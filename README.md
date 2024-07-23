Code for Our ICML 2024 paper [Characterizing Truthfulness in Large Language Model Generations with Local Intrinsic Dimension](https://arxiv.org/abs/2402.18048).

## Environment
Use the .yml file:

<code>conda env create -f lid.yml</code>

The code is tested under python3.9 and CUDA12.3

## Run the code
We experimented with Llama-2-7B, Llama-2-13B, and Mistral-7B-v0.1. Please acquire the access through huggingface hub.

We reported results on the following datasets. We pick 2,000 examples to test from each dataset. 500 correctly answered queries and their answers are used as reference points:

[TriviaQA](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

[TydiQA (English)](https://huggingface.co/datasets/google-research-datasets/tydiqa)

[CoQA](https://stanfordnlp.github.io/coqa/)

[HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) 

You first need to generate the responses for queries in a dataset. The queries and responses are written in a csv file. The format is demonstrated in the <code>assert</code> directory.
Then, run <code>bash run_pipeline.sh</code>, which contains to two steps:

1. Obtain the representations for last tokens in the responses for all layers. Write the tensors into a directory

2. Run the LID-MLE estimator to obtain the LIDs and AUROC for untruthfulness detection.

