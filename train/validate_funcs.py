
from transformers import T5ForConditionalGeneration, RobertaTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_metric, load_from_disk, load_dataset
import numpy as np
import pandas as pd
import evaluate
import torch
device = "cuda" if torch.cuda.is_available() else 'cpu'

base_model = "Pipper/SolCoderFuncs"
metric = evaluate.load('rouge')
tokenizer = RobertaTokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model).to(device)

small_ds = dataset = load_dataset("Pipper/SolFuncs")['test']
#small_ds = dataset.select(np.arange(100,5000,10))

max_input_length = 150
max_target_length = 256

def strip_comment(com):
    com = com.replace('*','').strip()
    com = com.replace('@title','').strip()
    com = com.replace('@author','').strip()
    com = com.replace('@notice','').strip()
    com = com.replace('@dev','').strip()
    com = com.replace('@param','').strip()
    com = com.replace('#','').strip()
    com = com.replace('@return','return').strip()
    return com

def process_samples(samples):
    codes = samples['code_string']
    comments = samples['comments']
    inputs = [strip_comment(cm) for cm in comments]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)
    # encode the summaries
    labels = tokenizer(codes, max_length=max_target_length, padding="max_length", truncation=True, return_overflowing_tokens=True).input_ids
    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)
    model_inputs["labels"] = labels_with_ignore_index
    return model_inputs

def compute_metrics_2(preds, labels):
    labels = np.where(np.array(labels) != -100, labels, tokenizer.pad_token_id)
    preds = np.where(np.array(preds) != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=[decoded_preds], references=[decoded_labels], use_stemmer=True)
    return result 

def infer(samples):
    comments = samples['comments']
    inputs = [strip_comment(cm) for cm in comments]
    model_inputs = tokenizer(list(inputs), max_length=max_input_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
    generated_ids = model.generate(model_inputs, max_new_tokens=max_target_length)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    samples['generated_text'] = generated_text
    return samples

def get_metric(sample):
    sample['rouge'] = metric.compute(predictions=[sample['generated_text']], references=[sample['code_string']], use_stemmer=True)
    return sample


small_ds = small_ds.map(process_samples, batched=True, batch_size=8, num_proc=1).map(infer)
small_ds = small_ds.map(get_metric, num_proc=20)

rouge_results = pd.DataFrame(small_ds['rouge'])

print('INFO: Mean Rouge Score')
print(rouge_results.mean(axis=0))