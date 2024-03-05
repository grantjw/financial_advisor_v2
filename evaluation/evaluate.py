from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, HfArgumentParser
from accelerate import Accelerator
from datasets import load_dataset
import torch
from peft import LoraConfig
from trl import SFTTrainer
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
# Define and parse arguments.
@dataclass
class ScriptArguments:
    # training parameters
    model_name: Optional[str] = field(
        default="results/dpo-finqa/final_model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    tokenizer_name: Optional[str] = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={"help": "The tokenizer to use"}
    )

    per_device_eval_batch_size: Optional[int] = field(default=4, metadata={"help": "eval batch size per device"})
    num_test_samples: Optional[int] = field(default=100, metadata={"help": "number of samples to evaluate on"})

    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})

    output_dir: Optional[str] = field(default="results/dpo-finqa-results", metadata={"help": "the output directory"})
    save_path: Optional[str] = field(default="evaluation/results.csv", metadata={"help": "where to save the losses to"})
    cache_dir: Optional[str] = field(default="cache_dir")

def create_dataset(args):
    dataset = load_dataset("gbharti/finance-alpaca", cache_dir=args.cache_dir, split="train[:24%]")
    # test_idx = range(int(len(dataset) * 0.8), len(dataset))
    test_idx = range(int(len(dataset) * 0.8), int(len(dataset) * 0.8) + args.num_test_samples)
    dataset = dataset.select(test_idx)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(
        lambda example: {"text": 
                        f"<s>[INST] {example['instruction']} [/INST] {example['output']}<\s>"},
        num_proc=4)
    return dataset


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Create results file if it doesn't already exist
    if not os.path.exists(script_args.save_path):
        with open(script_args.save_path, "w") as f:
            f.write("model_name,loss\n")

    # Model setup
    tokenizer = AutoTokenizer.from_pretrained(script_args.tokenizer_name,
                                              cache_dir=script_args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        use_nested_quant = True,
    )

    model = AutoModelForCausalLM.from_pretrained(script_args.model_name, 
                                                    quantization_config=bnb_config,
                                                    device_map={"": Accelerator().local_process_index},
                                                    cache_dir=script_args.cache_dir)

    dataset = create_dataset(script_args)
    
    training_args = TrainingArguments(
        output_dir="evaluation/dpo-finqa-results",
        per_device_eval_batch_size=4,
        report_to="none"
    )

    peft_config = LoraConfig(
                    r=32,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    target_modules=(
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                        "lm_head",),
                    bias="none",
                    task_type="CAUSAL_LM")

    trainer = SFTTrainer(
        model=model,
        dataset_text_field="text",
        train_dataset=dataset,
        eval_dataset=dataset,
        peft_config=peft_config,
        packing=True,
        max_seq_length=script_args.max_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    metrics = trainer.evaluate()
    print(metrics)

    with open(script_args.save_path, "a") as f:
        f.write(f"{script_args.model_name},{metrics['eval_loss']}\n")
            

