import argparse
import os
import torch
from peft import AutoPeftModelForCausalLM

if __name__ == "__main__":
    # Free memory for merging weights
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", help="The file to load the checkpoint from")
    parser.add_argument("--merged_path", help="The path to save the merged model to")
    parser.add_argument("--cache_dir", default="cache_dir")
    args = parser.parse_args()

    torch.cuda.empty_cache()
    print("Loading model...")
    model = AutoPeftModelForCausalLM.from_pretrained(args.checkpoint_path, 
                                                     device_map="auto", 
                                                     torch_dtype=torch.float16, 
                                                     cache_dir=args.cache_dir)
    print("Merging model...")
    model = model.merge_and_unload()
    print("Saving model...")
    model.save_pretrained(args.merged_path, safe_serialization=True)
    print("Done!")