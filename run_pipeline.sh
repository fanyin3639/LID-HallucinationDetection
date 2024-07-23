export CUDA_VISIBLE_DEVICES=3,4
export HF_HOME="provide your huggingface home here"
source ~/.bashrc

# Initialize Conda environment
eval "$(conda shell.bash hook)"

conda activate lid
python src/get_activation.py \
	--data_name coqa \
	--model_name  mistralai/Mistral-7B-v0.1 \
	--dataset_name ./asset/mistral-7b/coqa/lid_5000_samples/results/prepared_data.csv \
	--dir_name ./asset/mistral-7b/coqa/lid_5000_samples/results/
