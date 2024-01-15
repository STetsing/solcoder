import os, sys
import datasets
import torch 
import os 
import numpy as np
import pandas as pd
import evaluate, math
from time import time
from datasets import Dataset
from tqdm import tqdm
from transformers.trainer import unwrap_model
from transformers import AutoConfig, AdamW, get_scheduler, T5ForConditionalGeneration, RobertaTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer, AutoTokenizer, DataCollatorForSeq2Seq
from accelerate import Accelerator
from datasets import load_metric, load_from_disk, load_dataset, DatasetDict
from datetime import datetime
from torch.utils.data import DataLoader

output_dir = './SolExplain'
checkpointing_steps = "epoch"
logger = print
num_train_epochs = 5 
#device = "cuda" 
accelerator = Accelerator()
device = Accelerator.device
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
tokenizer = RobertaTokenizer.from_pretrained(base_model)
config = AutoConfig.from_pretrained(base_model)
model = T5ForConditionalGeneration.from_pretrained(base_model, config=config)

max_input_length = 256
max_target_length = 100
prefix = "Generate Solidity: "

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

    # encode the summaries
    model_inputs = tokenizer(codes, max_length=max_input_length, padding="max_length", truncation=True)
    

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)
    
    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs

#if  accelerator.is_local_main_process:
with accelerator.main_process_first():
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
            
            # also push dataset to repo
            # split_dataset.push_to_hub("Pipper/sol_processed_s2s", token=os.environ.get("HF_TOKEN"))
            # print('Info: Pushed preprocessed data to hub')

        else:
            print('INFO: loading preprocessed set from disk...')
            dataset = load_from_disk('./sol_funcs', keep_in_memory=True)
            print('INFO: loaded preprocessed set from disk!')
    else: 
        print('INFO: loading preprocessed set from hugginface space...')
        dataset = load_dataset("Pipper/SolFuncs")
        print('INFO: loaded preprocessed set from hugginface space!')
        dataset = dataset.map(process_samples, batched=True, batch_size=8, num_proc=50)
        dataset = dataset.remove_columns(['file_name', 'comments', 'code_string', '__index_level_0__'])
        print('INFO: Test tokenized inputs')
        print(dataset) 
        print(tokenizer.decode(dataset['train'][0]['input_ids'], skip_special_tokens=True))
        preds = dataset['train'][0]['labels']
        preds = np.where(np.array(preds) != -100, preds, tokenizer.pad_token_id)
        print(tokenizer.decode(preds, skip_special_tokens=True))

        
        # dataset = dataset['train'].train_test_split(test_size=0.1, seed=100)
        # test_valid = dataset['test'].train_test_split(test_size=0.5, seed=100)

        # dataset = DatasetDict({
        #                                 'train': dataset['train'],
        #                                 'test': test_valid['test'],
        #                                 'valid': test_valid['train']
        #                                 })


        #dataset.push_to_hub("Pipper/sol_processed_s2s", token=os.environ.get("HF_TOKEN"), branch='cleaned', max_shard_size="1G")
        #print('INFO: pushed data to the hub')
        #sys.exit(1)

per_device_train_batch_size = 20
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
train_dataloader = DataLoader(dataset['train'], shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size)
eval_dataloader = DataLoader(dataset['valid'], shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size)


no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

#optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-4)
gradient_accumulation_steps = 1
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
max_train_steps = num_train_epochs * num_update_steps_per_epoch

# lr_scheduler = get_scheduler(
#         name="polynomial",
#         optimizer=optimizer,
#         num_warmup_steps=0,
#         num_training_steps=max_train_steps,
#     )

model, train_dataloader, eval_dataloader = accelerator.prepare(
        model, train_dataloader, eval_dataloader)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
max_train_steps = num_train_epochs * num_update_steps_per_epoch

num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
checkpointing_steps = None

total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps


logger("***** Running training *****")
logger(f"  Num examples = {len(dataset)}")
logger(f"  Num Epochs = {num_train_epochs}")
logger(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
logger(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
logger(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
logger(f"  Total optimization steps = {max_train_steps}")
# Only show the progress bar once on each machine.
progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
starting_epoch = 0
with_tracking = True

for epoch in range(starting_epoch, num_train_epochs):
    start_time = time()
    model.train()
    if with_tracking:
        total_loss = 0
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        # We keep track of the loss at each epoch
        if with_tracking:
            total_loss += loss.detach().float()
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)

        if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            # optimizer.step()
            # lr_scheduler.step()
            # optimizer.zero_grad()
            progress_bar.update(1)
            completed_steps += 1

        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_d = f"step_{completed_steps }"
                if output_dir is not None:
                    output_dir = os.path.join(output_dir, output_d)
                accelerator.save_state(output_dir)

        if completed_steps >= max_train_steps:
            break

    end_time = time()
    logger(f"Epoch {epoch} training took {end_time-start_time} seconds")
    start_time = time()
    model.eval()
    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1) if not is_regression else outputs.logits.squeeze()
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )
    end_time = time()
    logger(f"Epoch {epoch} evaluation took {end_time-start_time} seconds")

    eval_metric = metric.compute()
    logger(f"epoch {epoch}: {eval_metric}")

    if with_tracking:
        accelerator.log(
            {
                "train_loss": total_loss.item() / len(train_dataloader),
                "epoch": epoch,
                "step": completed_steps,
            },
            step=completed_steps,
        )

    

    if checkpointing_steps == "epoch":
        output_d = f"epoch_{epoch}"
        if output_dir is not None:
            output_dir = os.path.join(output_dir, output_d)
        accelerator.save_state(output_dir)

if output_dir is not None:
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )
    
if output_dir is not None:
    with open(os.path.join(output_dir, "all_results.json"), "w") as f:
        json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)
