import os 
import datasets
import torch 
import os 
import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, RobertaTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from accelerate import Accelerator
from datasets import load_metric, load_from_disk, load_dataset
from datetime import datetime

#device = "cuda" 

device = Accelerator.device
print('Info: Computing device is:', device)

metric = evaluate.load('rouge')
process_local = False

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(tokenizer(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(tokenizer(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result



base_model = "Salesforce/codet5-base"
sol_tok_model = "Pipper/finetuned_sol"
tokenizer = RobertaTokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model)

max_input_length = 512
max_target_length = 1024
prefix = "Generate Solidity: "

def process_samples(samples):
    codes = samples['code_string']
    comments = samples['comments']

    inputs = [prefix + cm for cm in comments]
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

if process_local:
    if not os.path.exists('./sol_dataset'):
        print('loading dataset for the first time')
        os.makedirs('./sol_dataset', exist_ok=True)
        data_path = 'filtered_comment_code_sol.pkl'
        df = pd.read_pickle(data_path)
        dataset = Dataset.from_pandas(df)
        dataset = dataset.map(process_samples, batched=True, batch_size=32, num_proc=8) 
        dataset.save_to_disk('./sol_dataset')
    else:
        print('Info: loading preprocessed set from disk ...')
        dataset = load_from_disk('./sol_dataset', keep_in_memory=True)
        print('Info: loaded preprocessed set from disk!')
else: 
    print('Info: loaded preprocessed set from hugginface space ...')
    dataset = load_dataset("Pipper/sol_processed_s2s", )
    print('Info: loaded preprocessed set from hugginface space!')

dataset = dataset.train_test_split(test_size=0.1)
train_set = dataset['train']
eval_set = dataset['test']

training_args = Seq2SeqTrainingArguments(
    "sol_processed_s2s",
    evaluation_strategy='epoch', 
    learning_rate=1e-4, 
    per_device_eval_batch_size=2,
    per_device_train_batch_size=2,
    num_train_epochs=30,
    push_to_hub=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    save_strategy = "epoch",
    prediction_loss_only=True,
    logging_strategy="steps",
    logging_steps=1000,
    seed=100
)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model, 
    tokenizer=tokenizer,
    args=training_args, 
    train_dataset=train_set, 
    eval_dataset=eval_set, 
    compute_metrics=compute_metrics,
    data_collator=data_collator # very important, does the label shifting by 1
)

trainer.train()
tokenizer.save_pretrained('./trained_model_last_epoch')
trainer.save_model('./trained_model_last_epoch')

trainer.push_to_hub(commit_message="training comment 2 code done"+datetime.now.strftime("%m/%d/%Y, %H:%M:%S"))