import os, sys
sys.path.append('./')
import numpy as np
import datasets
import evaluate
import math
import torch
import pandas as pd
from datasets import Dataset
from datasets import load_metric, load_from_disk 
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, T5ForConditionalGeneration
from transformers import DataCollatorForLanguageModeling
from accelerate import Accelerator
from utils.get_tokens_causal import *

accelerator = Accelerator()
device = accelerator.device
print('INFO: Computing device is:', device)
print('INFO: Tokenizer is fast:', tokenizer.is_fast)
torch.cuda.empty_cache()
data_dir = './SolCausal'

print('INFO: Loading model ...')
model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", trust_remote_code=True)
print('INFO: Model size is', model.num_parameters()/1e9, "GB\n")

dataset = load_from_disk(data_dir) 
dataset = dataset.map(process_samples,batched=True, num_proc=30, batch_size=100, remove_columns=dataset["train"].column_names)
dataset = dataset.map(group_texts,batch_size=50, batched=True, num_proc=30)

# print(tokenizer.decode(dataset['train']['input_ids'][10]))
# print('#'*100)
# print(tokenizer.decode(dataset['train']['input_ids'][11]))
# print('#'*100)
# print(tokenizer.decode(dataset['train']['input_ids'][12]))
# print(len(dataset['train']['input_ids'][0]))
print("INFO: Length dataset:",len(dataset))
print(dataset)

#print('INFO: Training shape:', np.array(dataset["train"]['input_ids']).shape)

training_args = TrainingArguments('Phi2-SolCoder', 
        evaluation_strategy="epoch", 
        learning_rate=1e-4, 
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        num_train_epochs=10,
        push_to_hub=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        gradient_accumulation_steps=32, 
        save_strategy = "epoch",
        prediction_loss_only=True,
        logging_strategy="steps",
        logging_steps=100,
        fp16=True,
        seed=100)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

class PerplexCallback(TrainerCallback):
    "A callback that computes the model perplexity"

    def on_epoch_end(self, args, state, control, **kwargs):
        eval_results = trainer.evaluate()
        print(f"\nModel Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,  mlm=False)
trainer = Trainer(
    model=model, 
    tokenizer=tokenizer,
    args=training_args, 
    train_dataset=dataset['train'], 
    eval_dataset=dataset['valid'], 
    callbacks=[PerplexCallback],
    #compute_metrics=compute_metrics,
    data_collator=data_collator # very important, does the label shifting by 1
)

trainer.train()

tokenizer.save_pretrained('./Phi2-SolCoder')
trainer.save_model('./Phi2-SolCoder')