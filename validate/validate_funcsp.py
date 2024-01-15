
from transformers import T5ForConditionalGeneration, RobertaTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_metric, load_from_disk, load_dataset
import numpy as np
import pandas as pd
import evaluate
from accelerate import Accelerator
import torch
import warnings
from tqdm import tqdm
from torch.utils.data import DataLoader
warnings.simplefilter("ignore")

#device = "cuda" if torch.cuda.is_available() else 'cpu'
accelerator = Accelerator()

with accelerator.main_process_first():
    base_model = "./SolExplain/checkpoint-52855"
    tokenizer = RobertaTokenizer.from_pretrained(base_model)
    model = T5ForConditionalGeneration.from_pretrained(base_model)#.to(device)

metric = evaluate.load('rouge', 'bleu')

small_ds = dataset = load_dataset("Pipper/SolFuncs")['test']
#small_ds = dataset.select(np.arange(100,5000,20))

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
    codes = [ "Explain this function: " + c for c in codes]
    labels = tokenizer(inputs, max_length=max_target_length, padding="max_length", truncation=True, return_overflowing_tokens=True).input_ids

    model_inputs = tokenizer(codes, max_length=max_input_length, padding="max_length", truncation=True)
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
    model_inputs = tokenizer(list(inputs), max_length=max_input_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=max_target_length)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    samples['generated_text'] = generated_text
    return samples

def get_metric(sample):
    sample['rouge'] = metric.compute(predictions=[sample['generated_text']], references=[sample['code_string']], use_stemmer=True)
    return sample


#small_ds = small_ds.map(process_samples, batched=True, batch_size=8metric.compute(predictions=[sample['generated_text']], references=[sample['code_string']], num_proc=32)
small_ds = small_ds.remove_columns("file_name")
small_ds = small_ds.remove_columns("__index_level_0__")
dataloader = DataLoader(small_ds, batch_size=200)

model, dataloader = accelerator.prepare(model, dataloader)
output_sequences = []
for batch in tqdm(dataloader):
    samples = batch
    with torch.inference_mode():
        codes = samples['code_string']
        codes = [ "Explain this function: " + c for c in codes]
        model_inputs = tokenizer(codes, max_length=max_input_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(accelerator.device)
        generated_ids = model.module.generate(model_inputs, max_new_tokens=max_target_length)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        output_sequences.extend([generated_text, samples['comments']])

accelerator.wait_for_everyone()
if  accelerator.is_local_main_process:
    preds = [x[0] for x in output_sequences]
    labels = [x[1] for x in output_sequences]
    print('#'*100)
    print(preds[0])
    print('#'*100)
    print(labels[0])
    print('#'*100)

    result = metric.compute(predictions=preds, references=labels)
    print(result)
# #.map(infer, batched=True, batch_size=200, num_proc=1)
# small_ds = small_ds.map(get_metric, num_proc=64)

# rouge_results = pd.DataFrame(small_ds['rouge'])

# print('INFO: Mean Rouge Score')
# print(rouge_results.mean(axis=0))