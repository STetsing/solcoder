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
from transformers import DataCollatorForLanguageModeling, BitsAndByteConfig, HfArgumentParser
from accelerate import Accelerator
from utils.get_tokens_causal import *
from peft import LoraConfig, prepare_model_for_kit_training
from trl import SFTTrainer

bnb_conig = BitsAndByteConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4", # normalized float
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

accelerator = Accelerator()
device = accelerator.device
print('INFO: Computing device is:', device)
print('INFO: Tokenizer is fast:', tokenizer.is_fast)
torch.cuda.empty_cache()
data_dir = './SolCausal'

print('INFO: Loading model ...')
model = AutoModelForCausalLM.from_pretrained(base_model, 
                    torch_dtype="auto", 
                    quantization_config = bnb_conig,
                    trust_remote_code=True)
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kit_training(model, use_gradient_checkpointing=True)

print('INFO: Model size is', model.num_parameters()/1e9, "GB\n")

dataset = load_from_disk(data_dir) 
dataset = dataset.map(process_sampl,batched=True, num_proc=None, remove_columns=dataset["train"].column_names,)\
                .map(group_texts, batched=True, num_proc=30)

# print(tokenizer.decode(dataset['train']['input_ids'][10]))
# print('#'*100)
# print(tokenizer.decode(dataset['train']['input_ids'][11]))
# print('#'*100)
# print(tokenizer.decode(dataset['train']['input_ids'][12]))
print(len(dataset['train']['input_ids'][0]))
print(len(dataset))

training_args = TrainingArguments('SolCoderNew', 
        evaluation_strategy="epoch", 
        learning_rate=2e-4, 
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        num_train_epochs=10,
        push_to_hub=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy = "epoch",
        prediction_loss_only=True,
        logging_strategy="steps",
        gradient_accumulation_steps=32, 
        eval_steps = 500,
        logging_steps=500,
        optim="paged_adamw_8bit",
        lr_scheduler_type = "cosine",
        warmupo_ratio = 0.05,
        weight_decay = 0.01
        fp16=True,
        seed=100)

perf_config = LoraConfig(
    r = 32, 
    lora_alpha = 64, 
    lora_dropout = 0.05,
    bias_type = None,
    task_type = "Causal_lm",
    target_modules = ["Wqkv", "fc1", "fc2" ]
)


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
trainer = SFTTrainer(
    model=model, 
    tokenizer=tokenizer,
    args=training_args, 
    train_dataset=dataset['train'], 
    eval_dataset=dataset['valid'], 
    callbacks=[PerplexCallback],
    peft_config = peft_config,
    #compute_metrics=compute_metrics,
    data_collator=data_collator # very important, does the label shifting by 1
)

trainer.train()

tokenizer.save_pretrained('./SolCoderNew')
trainer.save_model('./SolCoderNew')
