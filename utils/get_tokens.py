from transformers import RobertaTokenizer, AutoTokenizer
import re

base_model = "Salesforce/codet5p-220m"
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.truncation_side = 'left'
tokenizer.padding_side = 'right'
max_input_length = 512
max_target_length = 256

def strip_comment(com):
    com = com.replace('*','').strip()
    com = com.replace('\n','').strip()
    com = com.replace('\t','').strip()
    com = com.replace('  ','').strip()
    com = com.replace('@title','').strip()
    com = com.replace('@author','').strip()
    com = com.replace('@notice','').strip()
    com = com.replace('@dev','').strip()
    com = com.replace('@param','').strip()
    com = com.replace('#','').strip()
    com = com.replace('@return','return').strip()
    if com.startswith('/'):
        com = '/' + com 
    com = ''.join(com).strip()
    return com

def process_samples(samples):
    codes = samples['code_string']
    context = [ctx + "\n" + strip_comment(com)for ctx, com in zip(samples['context'], samples['comments'])]

    model_inputs = tokenizer(context, max_length=max_input_length, padding="max_length", truncation=True)
    labels = tokenizer(codes, max_length=max_target_length, padding="max_length", truncation=False, return_overflowing_tokens=True).input_ids

    # important: we need to replace the index of the padding tokens by -100
    # such that they are not taken into account by the CrossEntropyLoss
    labels_with_ignore_index = []
    for labels_example in labels:
        labels_example = [label if label != 0 else -100 for label in labels_example]
        labels_with_ignore_index.append(labels_example)
    
    model_inputs["labels"] = labels_with_ignore_index

    return model_inputs