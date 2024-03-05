import argparse
import os
import pandas as pd
from tqdm import tqdm
import csv

from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset

import torch
from torch.utils.data import DataLoader

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        help="the model to generate responses from")
    parser.add_argument("--tokenizer_name",
                        default=None,
                        help="the tokenizer's name")
    parser.add_argument("--dataset_name",
                        default="gbharti/finance-alpaca",
                        help="the dataset to generate responses to")
    parser.add_argument("--cache_dir",
                        default="cache_dir",
                        help="where to cache the huggingface objects")
    parser.add_argument("--save_path",
                        default="feedback/unlabeled.csv",
                        help="path to save the unlabeled data to")
    parser.add_argument("--max_len",
                        type=int,
                        default=1024,
                        help="max length of output")
    parser.add_argument("--batch_size",
                        type=int,
                        default=4,
                        help="number of prompts to generate responses for at once")
    parser.add_argument("--num_steps",
                        type=int,
                        default=-1,
                        help="number of batches to save. -1 means do not stop.")
    parser.add_argument("--write_steps",
                        type=int,
                        default=5,
                        help="number of batches to iterate through before saving as a chunk")
    args = parser.parse_args()
    return args


def load_finance_dataset(save_path, cache_dir):
    # Read to the correct number in the dataset
    df = pd.read_csv(save_path, header=0)
    ds = load_dataset("gbharti/finance-alpaca", cache_dir=cache_dir, split=f"train[:24%]")
    idxs = range(int(0.6 * len(ds)) + len(df), int(0.8 * len(ds)))
    ds = ds.select(idxs)
    ds = ds.map(
        lambda example: {"text": 
                        f"<s>[INST] {example['instruction']} [/INST]"},
        num_proc=4)
    return ds


def generate_response_pair(model, tokenizer, batch, max_len):
    """
    Generates a pair of responses given a batch of prompts, a model, and a tokenizer.
    Input should be shape N list of strings
    Output is 3 x N lists of strings
    """
    tokens = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
    prompt_len = tokens.input_ids.shape[1]
    out_a = model.generate(**tokens, max_length=max_len, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    out_b = model.generate(**tokens, max_length=max_len, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    prompts = tokenizer.batch_decode(out_a[:,:prompt_len], skip_special_tokens=True) 
    responses_a = tokenizer.batch_decode(out_a[:,prompt_len:], skip_special_tokens=True)
    responses_b = tokenizer.batch_decode(out_b[:,prompt_len:], skip_special_tokens=True)
    return prompts, responses_a, responses_b


if __name__=="__main__":
    args = create_argparser()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype="float16",
        use_nested_quant = True,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_name, 
                                                 quantization_config=bnb_config,
                                                 device_map={"": Accelerator().local_process_index},
                                                 cache_dir=args.cache_dir)
    model.eval()
    tokenizer_name = args.tokenizer_name if args.tokenizer_name else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, padding_side="left", cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # If the file doesn't exist, create the header
    if not os.path.exists(args.save_path):
        with(open(args.save_path, "w")) as f:
            f.write("prompt,response_a,response_b\n")

    ds = None
    if args.dataset_name == "gbharti/finance-alpaca":
        ds = load_finance_dataset(args.save_path, args.cache_dir)
    else:
        raise ValueError("Invalid value for args.dataset_name")
    
    with open(args.save_path, "a", newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        write_batch = []
        dl = DataLoader(ds["text"], batch_size=args.batch_size, shuffle=False)
        for step, batch in tqdm(enumerate(dl), total=args.num_steps):
            if step == args.num_steps:
                break

            # Generate responses for batch
            prompts, responses_a, responses_b = generate_response_pair(model, tokenizer, batch, args.max_len)

            # Batch out outputs so we minimize the amount of file writes.
            # Lines are wrapped in quotes so the commas can go in the csv.
            write_batch += list(zip(prompts, responses_a, responses_b))
            if (step + 1) % args.write_steps == 0:
                writer.writerows(write_batch)
                write_batch.clear()

        writer.writerows(write_batch)