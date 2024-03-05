# fin-rlhf

Run SFT with: 
```
accelerate launch sft.py
```

Merge SFT LORA weights with 
```
python merge.py --checkpoint_path="results/sft-finqa/checkpoint-550" --merged_path="results/sft-finqa/final_model
```

Generate dataset with 
```
accelerate launch feedback/generate.py --model_name="results/sft-finqa/final_model" --tokenizer_name="mistralai/Mistral-7B-v0.1" --dataset_name="gbharti/finance-alpaca" --save_path="feedback/finance-alpaca-unlabeled.csv" --num_steps=100
```

Annotate dataset with 
```
python feedback/annotate.py --unlabeled_path="feedback/finance-alpaca-unlabeled.csv" --labels_path="feedback/finance-alpaca-labels.csv"
```

Merge generations with annotations and upload to huggingface with 
```
python feedback/merge_labels.py --hf_repo="danyoung/finance-feedback"
```

Run DPO with
```
accelerate launch dpo.py
```
(Or, you can run "reward_model.py" and "ppo_finqa.ipynb". We met CUDA out of memory issues. The code was not runnable on TR4 8 vCPU 30 GB RAM 16GB VRAM)

Upload model with 
```
huggingface-cli upload danyoung/finance-qa results/sft-finqa/final_model
```

Evaluate a model with
```
accelerate launch evaluation/evaluate.py --model_name="results/sft-finqa/final_model" --tokenizer_name="mistralai/Mistral-7B-v0.1"
```
#### Python environment
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

#### Data
```
python3 ingest.py
```
#### Environment Variables
```
cp example.env .env # update necessary fields
```

#### Run streamlit app
```
streamlit run app.py
```