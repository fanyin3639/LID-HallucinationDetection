# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional
import re
import os
import ast
import string
import nltk
import torch
import torch.nn as nn
import evaluate
import pandas as pd
from metrics import f1_score, rouge_L
from accelerate import Accelerator
from datasets import load_dataset, Dataset
# from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from torch.utils.data import DataLoader
from accelerate import infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch, load_checkpoint_in_model, dispatch_model

tqdm.pandas()


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    model_name: Optional[str] = field(default="/local2/fanyin/Llama-2-13b-hf", metadata={"help": "the model name"})
    data_name: Optional[str] = field(default="triviaqa", metadata={"help": "the model name"})
    dataset_name: Optional[str] = field(
        default="/local2/fanyin/icl-selective-annotation/outputs/llama-2-7b/tydiqa/lid_4000_samples_0_context/results/prepared_data.csv", metadata={"help": "the dataset name"}
    )
    model_cache_dir: Optional[str] = field(
        default="/local1/fanyin/models", metadata={"help": "the dataset name"}
    )

    dir_name: Optional[str] = field(
        default="/local1/fanyin/icl-selective-annotation/outputs/llama-2-7b/tydiqa/lid_4000_samples_0_context/results/", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default="none", metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=2e-5, metadata={"help": "the learning rate"})
    batch_size: Optional[int] = field(default=2, metadata={"help": "the batch size"})
    warmup_steps: Optional[int] = field(default=100, metadata={"help": "the warmup step"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    evaluation_strategy: Optional[str] = field(default='epoch', metadata={"help": "Evaluation strategy"})
    local_rank: Optional[int] = field(default=-1, metadata={"help": "local rank"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=False, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    peft_lora_r: Optional[int] = field(default=64, metadata={"help": "the r parameter of the LoRA adapters"})
    peft_lora_alpha: Optional[int] = field(default=16, metadata={"help": "the alpha parameter of the LoRA adapters"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the number of logging steps"})
    use_auth_token: Optional[bool] = field(default=False, metadata={"help": "Use HF auth token to access the model"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    max_steps: Optional[int] = field(default=-1, metadata={"help": "the number of training steps"})
    save_steps: Optional[int] = field(
        default=500, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    only_question: Optional[bool] = field(default=False, metadata={"help": "Only use question"})
    save_total_limit: Optional[int] = field(default=10, metadata={"help": "Limits total number of checkpoints."})
    push_to_hub: Optional[bool] = field(default=False, metadata={"help": "Push the model to HF Hub"})
    hub_model_id: Optional[str] = field(default=None, metadata={"help": "The name of the model on HF Hub"})
    # deepspeed_file: Optional[str] = field(default='/local2/fanyin/icl-selective-annotation/trl/trl/stage3.config', metadata={"help": "The name of the model on HF Hub"})





def construct_new_output(context, output, tokenizer, model):
    def clean_generated_text(generated_texts):
        stop = ['--', '</s>', '<unk>', '\n', ';', '#', "Q:"]
        prediction = []
        for idx, generated_text in enumerate(generated_texts):
            stop_index = len(generated_text)
            for i, c in enumerate(generated_text):
                if c.strip(' ') in stop:
                    stop_index = i
                    break
            generated_text.replace('1. ', '')
            sg = generated_text[:stop_index].replace('</s>', '')
            prediction.append(sg)
        return prediction

    output_tokens = output.split()
    print(output_tokens)
    if len(output_tokens) == 1:
        input = tokenizer(context, return_tensors="pt")
        gen_tokens = model.generate(
            input['input_ids'].to(0),
            do_sample=True,
            max_length=input['input_ids'].shape[1] + 10,
            use_cache=True,
            top_p=0.9,
            top_k=0,
            num_return_sequences=10,
            output_scores=True, return_dict_in_generate=True
        )
        print(construct_input(context, output))
        generations = tokenizer.batch_decode(gen_tokens.sequences[:, len(input['input_ids'][0]):], skip_special_tokens=True)
        generations = clean_generated_text(generations)
        print(generations)
        print(construct_input(context, generations[0]))
    else:
        new_output = ' '.join(output_tokens[:-1])
        input = tokenizer(construct_input(context, new_output), return_tensors="pt")
        gen_tokens = model.generate(
            input['input_ids'].to(0),
            do_sample=True,
            max_length=input['input_ids'].shape[1] + 10,
            use_cache=True,
            top_p=0.9,
            top_k=0,
            num_return_sequences=10,
            output_scores=True, return_dict_in_generate=True
        )
        print(construct_input(context, output))
        generations = tokenizer.batch_decode(gen_tokens.sequences[:, len(input['input_ids'][0]):], skip_special_tokens=True)
        generations = clean_generated_text(generations)
        print(generations)
        print(construct_input(construct_input(context, new_output) + ' ', generations[0]))
    print()
    return generations

def construct_input(context, answer):
    # 'Consider the amount of truthfulness in the following answer:' +
    inputs = context + answer
    return inputs


def expand_df_to_data(script_args, df, labels):
    expanded_df = {'input': [], 'label': []}
    eval_expanded_df = {'input': [], 'label': [], 'gt': [], 'prefix': []}

    cnt = 0
    for idx, row in tqdm(df.iterrows(), total=len(list(df.iterrows()))):
        preds = ast.literal_eval(row['sampled_answers'])
        preds = [p for p in preds if not len(p.strip()) == 0]

        if len(preds) == 0:
            continue
        golds = ast.literal_eval(row['gold_answer'])

        rouge_scores = rouge_L(preds[:1], golds[:1])
        batch = [construct_input(row['context'], pred) for pred in preds[:1]]
        eval_expanded_df['input'].append(batch)

        eval_expanded_df['prefix'].append(row['context'])
        eval_expanded_df['label'].append(rouge_scores > 0.5)
        eval_expanded_df['gt'].append([construct_input(row['context'], golds[0])])

        cnt += 1

    return pd.DataFrame(expanded_df), pd.DataFrame(eval_expanded_df)

class Hook:
    def __init__(self):
        self.out = None

    def __call__(self, module, module_inputs, module_outputs):
        self.out, _ = module_outputs

def get_acts(example, tokenizer, model, layers, device, return_all=True, prefix=None):
    """
    Get given layer activations for the statements.
    Return dictionary of stacked activations.
    """
    # attach hooks
    hooks, handles = [], []
    for layer in layers:
        hook = Hook()
        handle = model.model.layers[layer].register_forward_hook(hook)
        hooks.append(hook), handles.append(handle)
    # get activations
    acts = {layer: [] for layer in layers}
    tokenized_input = tokenizer(example, return_tensors="pt", padding=True)

    input_ids = tokenized_input['input_ids'].to(device)
    attention_masks = tokenized_input['attention_mask'].to(device)

    length = 0
    if return_all or prefix is not None:
        tokenized_input_prefix = tokenizer(prefix, return_tensors="pt", padding=False)
        prefix_length = tokenized_input_prefix['input_ids'][0].shape[0]
        length = attention_masks.sum(dim=-1) - prefix_length

    labels = input_ids.clone()
    labels = attention_masks.int() * labels + (1 - attention_masks.int()) * -100

    model_output = model(input_ids, attention_mask=attention_masks, labels=labels, output_hidden_states=True)
    shift_logits = model_output.logits[..., :-1, :].contiguous()

    shift_labels = labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    loss = loss_fct(shift_logits.permute(0,2,1), shift_labels)
    loss = loss.sum(-1).cpu().numpy().tolist()
    for layer, hook in zip(layers, hooks):
        if not return_all:
            acts[layer].append(hook.out[torch.arange(hook.out.shape[0]), attention_masks.sum(dim=-1).cpu() - 1])
        else:
            acts[layer].append(hook.out[:, prefix_length:])
    for layer, act in acts.items():
        acts[layer] = torch.stack(act)[0, :].cpu().float()
    # remove hooks
    for handle in handles:
        handle.remove()

    return loss, acts, length

# Step 3: Load the model

def load_model(script_args):
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        torch_dtype = None
    torch.set_grad_enabled(False)


    NO_SPLIT_MODULE_CLASSES = {"mistralai/Mistral-7B-v0.1": "MistralDecoderLayer", "facebook/opt-6.7B": "OPTDecoderLayer", "facebook/opt-13B": "OPTDecoderLayer", "EleutherAI/gpt-j-6B": "GPTJBlock", '/local2/fanyin/Llama-2-7b-hf': "LlamaDecoderLayer", '/local2/fanyin/Llama-2-13b-hf': "LlamaDecoderLayer", "/local2/fanyin/llama-7B": "LlamaDecoderLayer",  "openlm-research/open_llama_7b_v2": "LlamaDecoderLayer", "openlm-research/open_llama_3b_v2": "LlamaDecoderLayer"}
    if script_args.model_name in NO_SPLIT_MODULE_CLASSES:
        no_split_module_classes = [NO_SPLIT_MODULE_CLASSES[script_args.model_name]]
    else:
        no_split_module_classes = "LlamaDecoderLayer"

    max_memory = {0: '15GiB', 1: '15GiB'}
    print(script_args.model_name)

    config = AutoConfig.from_pretrained(script_args.model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)

    model = AutoModelForCausalLM.from_pretrained(script_args.model_name, device_map=device_map,
                                                           max_memory=max_memory, cache_dir=script_args.model_cache_dir)

    num_layers = len(model.model.layers)

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer, num_layers



def main():
    # Load arguments
    if not os.path.exists('output_tensors'):
        os.mkdir('output_tensors')
    # Preserve the trl trainer arguments
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model_name_base = os.path.basename(script_args.model_name)
    # Load model, tokenizer
    model, tokenizer, num_layers = load_model(script_args)


    # Load data
    df = pd.read_csv(script_args.dataset_name)

    # One can load pre-written labels for this task for efficiency, but computing labels everytime when load dataset is recommanded
    labels = pd.read_csv(os.path.join(script_args.dir_name, 'labels.csv'))

    df, eval_df = expand_df_to_data(script_args, df, labels)

    eval_dataset = Dataset.from_pandas(eval_df)
    dataloader = DataLoader(eval_dataset, batch_size=1)

    # Start extracting activations
    layers = [i for i in range(num_layers)]
    pred_acts = {l: [] for l in layers}
    gt_acts = {l: [] for l in layers}
    labels = []
    cnt = 0
    for batch in tqdm(dataloader, total=len(dataloader)):
        batch['input'] = [i[0] for i in batch['input']]
        batch['gt'] = [i[0] for i in batch['gt']]

        #### forward pass for getting all elements
        loss, pred_act, length = get_acts(batch['input'], tokenizer, model, layers, 'cuda:0', return_all=False, prefix=batch['prefix'])
        _, gt_act, _ = get_acts(batch['gt'], tokenizer, model, layers, 'cuda:0', return_all=False, prefix=batch['prefix'])

        #### extract each layer of features
        for layer in layers:
            pred_acts[layer].append(pred_act[layer][0, :])
            gt_acts[layer].append(gt_act[layer][0, :])

        #### get all other features
        labels.append(batch['label'])

    # Write to disk based on layers
    for layer in layers:
        torch.save(torch.stack(gt_acts[layer]), f"output_tensors/{model_name_base}_{script_args.data_name}_all_layer_{layer}_gt.pt")
        torch.save(pred_acts[layer], f"output_tensors/{model_name_base}_{script_args.data_name}_all_layer_{layer}_pred.pt")
        torch.save(torch.tensor(labels), f"output_tensors/{model_name_base}_{script_args.data_name}_all_layer_{layer}_label.pt")


if __name__ == '__main__':
    main()
