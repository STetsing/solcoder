from transformers import RobertaTokenizer, AutoTokenizer
import re

base_model = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
block_size = 64

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Calculate the total number of blocks
    num_blocks = (total_length + block_size - 1) // block_size
    # Pad the concatenated examples with eos if necessary
    pad_amount = num_blocks * block_size - total_length
    for k in concatenated_examples.keys():
        concatenated_examples[k] += [0] * pad_amount
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

def process_sampl(samples):
    return tokenizer(samples["source_code"])