import pandas as pd
import re
# remove lines with no code
def hasMarker(code):
    if '}' in code:
        return True
    
    elif '{' in code:
        return True
    
    else:
        return False


def has_no_license(code):
    return False if 'SPDX-License' in code else True

def discard_contract_or_lib(code):
    if "contract" in code:
        return False 

    if "library" in code:
        return False
    return True 

# remove docstring 
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

def remove_comments_from_context(code):
    # Regular expression to match Solidity comments (including nested comments)
    comment_pattern = re.compile(r'\/\/.*|\/\*[\s\S]*?\*\/')
    # Remove comments from the code
    while re.search(comment_pattern, code):
        code = re.sub(comment_pattern, '', code)

    return code


def apply_filter(dataset):
    print("INFO: Filtering Original dataset length:", len(dataset))
    dataset=dataset[dataset['code_string'].str.len() >= 20]
    dataset=dataset[dataset['comments'].str.len() >= 20]
    dataset=dataset[dataset['code_string'].apply(lambda x: hasMarker(x))]
    dataset=dataset[dataset['code_string'].apply(lambda x: has_no_license(x))]
    dataset = dataset.drop_duplicates(subset=['code_string'], keep='first')
    dataset = dataset.drop_duplicates(subset=['comments'], keep='first')
    dataset['comments']=dataset['comments'].apply(lambda x: strip_comment(x))
    print("INFO: Filtering dataset after all filters:", len(dataset))
    return dataset

