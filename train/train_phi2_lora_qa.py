import os, sys
sys.path.append('./')
import numpy as np
import datasets
import evaluate
import math
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from datasets import load_metric, load_from_disk, load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, Trainer, TrainerCallback, TrainingArguments, T5ForConditionalGeneration
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig, HfArgumentParser
from accelerate import Accelerator
from utils.get_tokens_causal import *
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# single GPU
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
peft_config = LoraConfig(
    r = 32, 
    lora_alpha = 64, 
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM",
    #target_modules = ["fc1", "fc2"]#, 'q_proj', 'k_proj', 'v_proj'] #,'dense','fc1','fc2',embed_tokens
    target_modules = ["fc1", "fc2", 'q_proj', 'k_proj', 'v_proj'] #,'dense','fc1','fc2',embed_tokens
)

bnb_conig = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type = "nf4", # normalized float
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

def print_trainable_parameters (model) :
    # Prints the number of trainable parameters in the model.
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters ( ):
        all_param += param .numel ( )
        if param. requires_grad:
            trainable_params += param. numel ()
    print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")



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
print(model)
print('INFO: Model size is', model.num_parameters()/1e9, "GB\n")


training_args = TrainingArguments('Phi2-SolCoder-lora-qa3', 
        evaluation_strategy="epoch", 
        learning_rate=2e-4, 
        per_device_eval_batch_size=14,
        per_device_train_batch_size=14,
        num_train_epochs=10,
        push_to_hub=True,
        save_total_limit=2,
        load_best_model_at_end=True,
        save_strategy = "epoch",
        prediction_loss_only=True,
        logging_strategy="steps",
        eval_steps = 100,
        logging_steps=100,
        optim="paged_adamw_8bit",
        gradient_accumulation_steps=2,
        lr_scheduler_type = "cosine",
        warmup_ratio = 0.05,
        weight_decay = 0.01,
        ddp_find_unused_parameters=False,
        push_to_hub_model_id="Phi2-SolCoder-lora-qa3",
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

response_template = "### Answer:"

data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

def formatting_prompts_func(example):
    text = f"### Solidity Instruction: {example['comments']}\n ### Answer: {example['code']}"
    return text 


# if not os.path.exists('./Solcoder_QA'):
#     data = pd.DataFrame()
#     for data_fln in tqdm(os.listdir('./data/'), desc='reading data'):
#         if '.csv' not in data_fln:
#             continue
#         else: 
#             content_df = pd.read_csv(os.path.join('./data/', data_fln))
#             data = pd.concat([data, content_df])

#     data['code'] = data['code_string']
#     dataset = Dataset.from_pandas(data)
#     train = dataset.train_test_split(test_size=0.2)
#     test_valid = train['test'].train_test_split(test_size=0.5)

#     dataset = DatasetDict({
#                             'train': train['train'],
#                             'test': test_valid['test'],
#                             'valid': test_valid['train']
#                             })
#     dataset.push_to_hub()

# else: 
#     dataset = load_from_disk('./Solcoder_QA', keep_in_memory=True)

dataset = load_dataset('Pipper/Solcoder_QA', keep_in_memory=True)
print('INFO: The dataset', dataset)
print("INFO: Length dataset:",len(dataset))
print(f"INFO: pocessing data on {os.cpu_count()} cores")

trainer = SFTTrainer(
    model=model, 
    tokenizer=tokenizer,
    args=training_args, 
    train_dataset=dataset['train'], 
    eval_dataset=dataset['valid'], 
    dataset_num_proc = os.cpu_count(),
    dataset_batch_size = 100,
    #dataset_text_field = 'code',
    #callbacks=[PerplexCallback],
    peft_config = peft_config,
    max_seq_length = 1024,
    formatting_func=formatting_prompts_func,
    packing=True,
    neftune_noise_alpha=5
    #compute_metrics=compute_metrics,
    #data_collator=data_collator # very important, does the label shifting by 1
)

print_trainable_parameters(model)


loader = trainer.get_train_dataloader()
for b in loader:
    break
print('Train batch shape is:', b['input_ids'].shape)
trainer.train()

tokenizer.save_pretrained('./Phi2-SolCoder-lora-qa3')
trainer.save_model('./Phi2-SolCoder-lora-qa3')