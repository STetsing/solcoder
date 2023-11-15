import torch 
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, RobertaTokenizer, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else 'cpu'

#model_path = './trained_model_last_epoch'
model_path = '/Users/pippertetsing/Desktop/work/Remix/solcoder/training_models/checkpoint-720'

tokenizer = RobertaTokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path).to(device)

def infer(comment):
    input_ids = tokenizer(comment, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(infer("function that adds 2 numbers"))