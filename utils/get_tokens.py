from transformers import RobertaTokenizer, AutoTokenizer
import re

base_model = "Salesforce/codet5p-220m"
tokenizer = RobertaTokenizer.from_pretrained(base_model)
max_input_length = 150
max_target_length = 256

def remove_comments_from_context(code):
    # Regular expression to match Solidity comments (including nested comments)
    comment_pattern = re.compile(r'\/\/.*|\/\*[\s\S]*?\*\/')
    # Remove comments from the code
    while re.search(comment_pattern, code):
        code = re.sub(comment_pattern, '', code)

    return code

def process_samples(samples):
    codes = samples['code_string']
    context_comment = [remove_comments_from_context(sp) for sp in samples['context']]

    model_inputs = tokenizer(context_comment, max_length=max_input_length, padding="max_length", truncation=True)

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