import os
import numpy as np 
from transformers import RobertaTokenizer, AutoTokenizer
from datasets import load_metric, load_from_disk, load_dataset, DatasetDict


base_model = "Pipper/SolCoder"
tokenizer = RobertaTokenizer.from_pretrained(base_model)

if not os.path.exists('./sol_dataset'):
    dataset = load_dataset("Pipper/sol_processed_s2s")
else:
    dataset = load_from_disk('./sol_dataset', keep_in_memory=True)

assert 'train' in dataset.keys()
assert 'valid' in dataset.keys()
assert 'test' in dataset.keys()

train_set = dataset['train']
eval_set = dataset['valid']
test_set = dataset['test']

# grap a random sample and print comment code pair
idx = 30
sample = train_set[idx]
print('Comment:', tokenizer.decode(sample['input_ids'], skip_special_tokens=True))
labels = np.where(np.array(sample['labels']) != -100, sample['labels'], tokenizer.pad_token_id)

print('code:\n', tokenizer.decode(labels, skip_special_tokens=True))

print('Verification succeed!')