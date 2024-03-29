import os, sys
sys.path.append('./')
import numpy as np
import datasets
import evaluate
import math
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import load_metric, load_from_disk 
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, T5ForConditionalGeneration
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig, HfArgumentParser
from accelerate import Accelerator
from utils.get_tokens_causal import *
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# single GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

bnb_conig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4", # normalized float
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

accelerator = Accelerator()
device = accelerator.device
print('INFO: Computing device is:', device)
print('INFO: Current device is:', torch.cuda.current_device())

print('INFO: Tokenizer is fast:', tokenizer.is_fast)
torch.cuda.empty_cache()
data_dir = './SolFuncsSmall'

print('INFO: Loading model ...')
model = AutoModelForCausalLM.from_pretrained(base_model, 
                    torch_dtype="auto", 
                    quantization_config = bnb_conig,
                    low_cpu_mem_usage=True,
                    load_in_4bit=True,
                    device_map={'':torch.cuda.current_device()},
                    trust_remote_code=True)
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})

print('INFO: Model size is', model.num_parameters()/1e9, "GB\n")

data_1 = pd.read_csv('./data/sourcify_0_comment_code_sol.csv')
data = data_1 # pd.concat([data_1, data_2, data_3])
data['code'] = data['code_string']
dataset = Dataset.from_pandas(data)
train = dataset.train_test_split(test_size=0.2)

dataset = DatasetDict({
                        'train': train['train'],
                        'valid': train['test']
                        })
print('INFO: The dataset', dataset)
#dataset['train'] = dataset['train'].select(np.arange(0, 10000, 1))
#dataset['valid'] = dataset['valid'].select(np.arange(0, 1000, 1))
#dataset['test'] = dataset['test'].select(np.arange(0, 1000, 1))
#dataset = dataset.map(process_samples, batched=True, num_proc=30, batch_size=100, remove_columns=dataset["train"].column_names)
#dataset = dataset.map(group_texts, batch_size=50, batched=True, num_proc=30)


# print(tokenizer.decode(dataset['train']['input_ids'][10]))
# print('#'*100)
# print(tokenizer.decode(dataset['train']['input_ids'][11]))
# print('#'*100)
# print(tokenizer.decode(dataset['train']['input_ids'][12]))
print("INFO: Length dataset:",len(dataset))
print(dataset)

def print_trainable_parameters (model) :
    # Prints the number of trainable parameters in the model.
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters ( ):
        all_param += param .numel ( )
        if param. requires_grad:
            trainable_params += param. numel ()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


training_args = TrainingArguments('Phi2-SolCoder-lora-qa2', 
        evaluation_strategy="epoch", 
        learning_rate=2e-4, 
        per_device_eval_batch_size=12,
        per_device_train_batch_size=12,
        num_train_epochs=10,
        push_to_hub=False,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy = "epoch",
        prediction_loss_only=True,
        logging_strategy="steps",
        eval_steps = 100,
        logging_steps=100,
        optim="paged_adamw_8bit",
        gradient_accumulation_steps=5,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.05,
        weight_decay = 0.01,
        ddp_find_unused_parameters=False,
        seed=100)

peft_config = LoraConfig(
    r = 16, 
    lora_alpha = 64, 
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
    target_modules = ["Wqkv", "fc1", "fc2"], # 'q_proj', 'k_proj', 'v_proj','dense','fc1','fc2',embed_tokens
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

response_template = " ### Answer:"

data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

def formatting_prompts_func(example):
    text = f"### Question: {example['comments']}\n ### Answer: {example['code']}"
    return text 

trainer = SFTTrainer(
    model=model, 
    tokenizer=tokenizer,
    args=training_args, 
    train_dataset=dataset['train'], 
    eval_dataset=dataset['valid'], 
    dataset_num_proc = 30,
    dataset_batch_size = 100,
    #dataset_text_field = 'code',
    #callbacks=[PerplexCallback],
    peft_config = peft_config,
    max_seq_length = 1024,
    formatting_func=formatting_prompts_func,
    packing=True,
    #compute_metrics=compute_metrics,
    #data_collator=data_collator # very important, does the label shifting by 1
)

print_trainable_parameters(model)
#trainer.train()

#tokenizer.save_pretrained('./Phi2-SolCoder-lora-qa2')
#trainer.save_model('./Phi2-SolCoder-lora-qa2')