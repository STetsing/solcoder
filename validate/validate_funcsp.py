
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

device = "cuda" if torch.cuda.is_available() else 'cpu'
accelerator = Accelerator()

with accelerator.main_process_first():
    base_model = "./SolExplain/checkpoint-287760/"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = T5ForConditionalGeneration.from_pretrained(base_model)#.to(device)

metric = evaluate.load('rouge')

small_ds = dataset = load_from_disk("./SolFuncsContext_60")['test']
#small_ds = dataset.select(np.arange(100,5000,20))

#print(len(small_ds['input_ids'][9]))

def compute_metrics_2(preds, labels):
    labels = np.where(np.array(labels) != -100, labels, tokenizer.pad_token_id)
    preds = np.where(np.array(preds) != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=[decoded_preds], references=[decoded_labels], use_stemmer=True)
    return result 

def infer(samples):
    comments = samples['comments']
    inputs = [cm for cm in comments]
    model_inputs = tokenizer(list(inputs), max_length=max_input_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(device)
    generated_ids = model.generate(model_inputs, max_new_tokens=max_target_length)
    generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    samples['generated_text'] = generated_text
    return samples

def get_metric(sample):
    sample['rouge'] = metric.compute(predictions=[sample['generated_text']], references=[sample['code_string']], use_stemmer=True)
    return sample


#small_ds = small_ds.map(process_samples, batched=True, batch_size=8metric.compute(predictions=[sample['generated_text']], references=[sample['code_string']], num_proc=32)
BS = 100
dataloader = DataLoader(small_ds, batch_size=BS)

model, dataloader = accelerator.prepare(model, dataloader)
output_sequences = []
for i in tqdm(range(0, len(small_ds), BS)):
    batch = samples = small_ds.select(np.arange(i,i+BS,1))
    with torch.inference_mode():
        codes = torch.IntTensor(samples['input_ids']).to(device)
        labels = np.where(np.array(samples['labels']) != -100, samples['labels'], tokenizer.pad_token_id)
        generated_ids = model.generate(codes)
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        labels = tokenizer.batch_decode(labels, skip_special_tokens=True, label_pad_token_id=-100)
        output_sequences.extend([generated_text, labels])

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