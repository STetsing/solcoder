import datasets
import torch 
import numpy as np
import pandas as pd
import evaluate
from datasets import Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, RobertaTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq

device = "cuda" if torch.cuda.is_available() else 'mps'

metric = evaluate.load('rouge')
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


data_path = 'filtered_comment_code_sol.pkl'
df = pd.read_pickle(data_path)[:100]
dataset = Dataset.from_pandas(df)

base_model = "Salesforce/codet5-base"
sol_tok_model = "Pipper/finetuned_sol"
tokenizer = RobertaTokenizer.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model).to(device)

max_input_length = 256
max_target_length = 512
prefix = "Generate Solidity: "

def process_samples(samples):
    codes = samples['code_string']
    comments = samples['comments']

    inputs = [prefix + cm for cm in comments]
    model_inputs = tokenizer(inputs, max_length=max_input_length, padding="max_length", truncation=True)

    # encode the summaries
    labels = tokenizer(codes, max_length=max_target_length, padding="max_length", truncation=False).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)
    
    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs

dataset = dataset.map(process_samples, batched=True, num_proc=5)
dataset = dataset.train_test_split(test_size=0.1)
train_set = dataset['train']
eval_set = dataset['test']

training_args = Seq2SeqTrainingArguments(
    "training_models",
    evaluation_strategy='epoch', 
    learning_rate=1e-4, 
    per_device_eval_batch_size=1,
    per_device_train_batch_size=1,
    num_train_epochs=10,
    push_to_hub=False,
    save_total_limit=2,
    load_best_model_at_end=True,
    save_strategy = "epoch",
    prediction_loss_only=True,
    logging_strategy="steps",
    logging_steps=500,
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
