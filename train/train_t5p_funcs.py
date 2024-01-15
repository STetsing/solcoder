import os, sys
import datasets
import torch 
import os 
import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset
from tqdm import tqdm
from transformers.trainer import unwrap_model
from transformers import T5ForConditionalGeneration, RobertaTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from accelerate import Accelerator
from datasets import load_metric, load_from_disk, load_dataset, DatasetDict
from datetime import datetime
from torch.utils.data import DataLoader

#device = "cuda" 
accelerator = Accelerator()
device = Accelerator.device
torch.cuda.empty_cache()
print('INFO: Computing device is:', device)

metric = evaluate.load('rouge')
process_local = False

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(np.array(labels) != -100, labels, tokenizer.pad_token_id)
    preds = np.where(np.array(preds) != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(tokenizer(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(tokenizer(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result


#base_model = "Salesforce/codet5p-770m"
base_model = "Salesforce/codet5p-220m"
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model)

if process_local:
    if not os.path.exists('./sol_funcs'):
        print('INFO: loading dataset for the first time ...')
        os.makedirs('./sol_funcs', exist_ok=True)
        data_path = 'filtered_comment_code_sol.pkl'
        df = pd.read_pickle(data_path)
        dataset = Dataset.from_pandas(df)
        del df
        dataset = dataset.map(process_samples, batched=True, batch_size=8, num_proc=15)
        dataset = dataset.train_test_split(test_size=0.1)
        test_valid = dataset['test'].train_test_split(test_size=0.5)

        split_dataset = DatasetDict({
                                    'train': dataset['train'],
                                    'test': test_valid['test'],
                                    'valid': test_valid['train']
                                    })

        split_dataset.save_to_disk('./sol_funcs')
        
    else:
        print('INFO: loading preprocessed set from disk...')
        dataset = load_from_disk('./sol_funcs', keep_in_memory=True)
        print('INFO: loaded preprocessed set from disk!')
else: 
    print('INFO: loading preprocessed set from hugginface space...')
    #dataset = load_dataset("Pipper/SolFuncs")
    dataset = load_from_disk("./SolFuncsContext_60")
    
    print('INFO: Test tokenized inputs')
    #print(dataset) 
    #print(tokenizer.decode(dataset['train'][0]['input_ids'], skip_special_tokens=True))
    preds = dataset['train'][0]['labels']
    preds = np.where(np.array(preds) != -100, preds, tokenizer.pad_token_id)
    #print(tokenizer.decode(preds, skip_special_tokens=True))


print(len(dataset['train']['input_ids'][0]))
print(dataset)
train_set = dataset['train']
eval_set = dataset['valid']


training_args = Seq2SeqTrainingArguments(
    "SolExplain",
    evaluation_strategy='epoch', 
    learning_rate=1e-4,
    per_device_eval_batch_size=10,
    per_device_train_batch_size=10,
    num_train_epochs=15,
    fp16=True,
    push_to_hub=True,
    save_total_limit=2,
    load_best_model_at_end=True,
    save_strategy = "epoch",
    prediction_loss_only=True,
    logging_strategy="steps",
    logging_steps=500,
    resume_from_checkpoint=True,
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

trainer.train(resume_from_checkpoint=True)
tokenizer.save_pretrained('./trained_model_last_epoch')
trainer.save_model('./trained_model_last_epoch')

#trainer.push_to_hub(commit_message="training latge t5 comment 2 code done "+datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))