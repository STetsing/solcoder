
from transformers import T5ForConditionalGeneration, RobertaTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from datasets import load_metric, load_from_disk, load_dataset
import numpy as np
import pandas as pd
import evaluate

base_model = "Pipper/sol_processed_s2s"
metric = evaluate.load('rouge')
tokenizer = RobertaTokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model)

dataset = load_dataset("Pipper/sol_processed_s2s")['train']
small_ds = dataset.select(np.arange(100,5000,10))

def compute_metrics_2(preds, labels):
    labels = np.where(np.array(labels) != -100, labels, tokenizer.pad_token_id)
    preds = np.where(np.array(preds) != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)
    result = metric.compute(predictions=[decoded_preds], references=[decoded_labels], use_stemmer=True)
    return result 

def infer(sample):
    comment = sample['comments']
    input_ids = tokenizer(comment, return_tensors='pt').input_ids
    outputs = model.generate(input_ids, max_new_tokens=200)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    sample['generated_text'] = generated_text
    return sample

def get_metric(sample):
    sample['rouge'] = metric.compute(predictions=[sample['generated_text']], references=[sample['code_string']], use_stemmer=True)
    return sample


small_ds = small_ds.map(infer)
small_ds = small_ds.map(get_metric)

rouge_results = pd.DataFrame(small_ds['rouge'])

print('Info: Rouge score')
print(rouge_results.mean(axis=0))