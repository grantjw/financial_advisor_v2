# 0. imports
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments, BitsAndBytesConfig

from trl import DPOTrainer

CACHE_DIR = "cache_dir/"

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})

    # training parameters
    train_pct: Optional[float] = field(default=1, metadata={"help": "the percent of data to train on"})
    model_name_or_path: Optional[str] = field(
        default="results/sft-finqa/final_model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    tokenizer_name: Optional[str] = field(
        default="mistralai/Mistral-7B-v0.1",
        metadata={"help": "The tokenizer to use"}
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    warmup_ratio: Optional[int] = field(default=0.03, metadata={"help": "the warmup ratio"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="paged_adamw_32bit", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=2, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=2, metadata={"help": "eval batch size per device"})
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "whether to use gradient checkpointing"}
    )

    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    max_prompt_length: Optional[int] = field(default=512, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=1024, metadata={"help": "the maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "number of epochs to train"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})
    save_steps: Optional[int] = field(default=5, metadata={"help": "the saving frequency"})
    eval_steps: Optional[int] = field(default=5, metadata={"help": "the evaluation frequency"})

    output_dir: Optional[str] = field(default="results/dpo-finqa-2", metadata={"help": "the output directory"})

    # instrumentation
    sanity_check: Optional[bool] = field(default=False, metadata={"help": "only train on 1000 samples"})
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )


def parse_to_dpo_format(sample):
    prompt = f"<s>{sample['prompt']} "
    chosen_key = "response_a" if sample["label"] == 0 else "response_b"
    rejected_key = "response_b" if sample["label"] == 0 else "response_a"
    chosen = f"{sample[chosen_key]}</s>"
    rejected = f"{sample[rejected_key]}</s>"
    return {
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    }

def create_datasets(args):
    dataset = load_dataset(
        "danyoung/finance-feedback",
        split="train",
        token=True,
        num_proc=4,
        cache_dir=CACHE_DIR,
    )
    # Filter out bad samples
    dataset = dataset.filter(lambda example: example["bad"] == 0)
    # Set up in nice format for dpo
    original_columns = dataset.column_names
    dataset = dataset.map(
        parse_to_dpo_format,
        num_proc=4,
        remove_columns=original_columns,
    )
    dataset = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42)
    train_data = dataset["train"]
    valid_data = dataset["test"]

    print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")
    return train_data, valid_data


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        use_nested_quant = True,
    )

    # 1. load a pretrained model
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        cache_dir=CACHE_DIR
    )
    model.config.use_cache = False

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        quantization_config=bnb_config,
        device_map={"": Accelerator().local_process_index},
        cache_dir=CACHE_DIR
    )
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name if script_args.tokenizer_name else script_args.model_name_or_path, 
        cache_dir=CACHE_DIR
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load the Stack-exchange paired dataset
    train_dataset, eval_dataset = create_datasets(script_args)

    # 4. initialize training arguments:
    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        eval_steps=script_args.eval_steps,
        output_dir=script_args.output_dir,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        warmup_ratio=script_args.warmup_ratio,
        optim=script_args.optimizer_type,
        remove_unused_columns=False,
        run_name="dpo-finqa",
    )

    peft_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "out_proj",
            "fc_in",
            "fc_out",
            "wte",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
    )

    # 6. train
    dpo_trainer.train()
    dpo_trainer.save_model(script_args.output_dir)

    # 7. save
    output_dir = os.path.join(script_args.output_dir, "final_unmerged")
    dpo_trainer.model.save_pretrained(output_dir)