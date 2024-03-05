import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, HfArgumentParser

from trl import SFTTrainer


@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="mistralai/Mistral-7B-v0.1", metadata={"help": "the model name"})

    resume_from_checkpoint: Optional[bool] = field(default=False, metadata={"help": "continue from a"})
    cache_dir: Optional[str] = field(default="cache_dir", metadata={"help": "where to cache huggingface objects"})
    dataset_name: Optional[str] = field(default="gbharti/finance-alpaca", metadata={"help": "the dataset name"})
    # subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train[:40%]", metadata={"help": "the split to use"})
    # size_valid_set: Optional[int] = field(default=100, metadata={"help": "the size of the validation set"})
    # streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    # shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    output_dir: Optional[str] = field(default="./results/sft-finqa")
    max_steps: Optional[int] = field(default=1000)
    logging_steps: Optional[int] = field(default=10)
    save_steps: Optional[int] = field(default=10)
    per_device_train_batch_size: Optional[int] = field(default=4)
    per_device_eval_batch_size: Optional[int] = field(default=4)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    gradient_checkpointing: Optional[bool] = field(default=True)
    group_by_length: Optional[bool] = field(default=False)
    max_grad_norm: Optional[float] = field(default=0.3)
    learning_rate: Optional[float] = field(default=2e-4)
    lr_scheduler_type: Optional[str] = field(default="constant")
    warmup_ratio: Optional[float] = field(default=0.03)
    weight_decay: Optional[float] = field(default=0.001)
    optim: Optional[str] = field(default="paged_adamw_32bit")
    run_name: Optional[str] = field(default="sft_finqa")
    report_to: Optional[str] = field(default="wandb")

    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})


def create_datasets(args):
    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        token=True,
        num_proc=args.num_workers,
        cache_dir=args.cache_dir,
    )
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.map(
        lambda example: {"text": f"<s>[INST] {example['instruction']} [/INST] {example['output']}</s>"}, 
        num_proc=args.num_workers
    )
    dataset = dataset.train_test_split(test_size=0.005, shuffle=False)
    train_data = dataset["train"]
    valid_data = dataset["test"]
    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    return train_data, valid_data

if __name__ == "__main__":

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    if script_args.group_by_length and script_args.packing:
        raise ValueError("Cannot use both packing and group by length")

    train_dataset, eval_dataset = create_datasets(script_args)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        use_nested_quant = True,
    )

    print(f"Accelerator process index: {Accelerator().local_process_index}")
    base_model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        cache_dir=script_args.cache_dir
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

    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, cache_dir=script_args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    training_args = TrainingArguments(
        output_dir=script_args.output_dir,
        max_steps=script_args.max_steps,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        group_by_length=script_args.group_by_length,
        max_grad_norm=script_args.max_grad_norm,
        learning_rate=script_args.learning_rate,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        weight_decay=script_args.weight_decay,
        optim=script_args.optim,
        run_name=script_args.run_name,
        report_to=script_args.report_to,
    )

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        packing=script_args.packing,
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()
    trainer.save_model(script_args.output_dir)

    output_dir = os.path.join(script_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

