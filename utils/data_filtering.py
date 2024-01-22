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
def rm_docstring(com):
    com = com.replace('*','').strip()
    com = com.replace('@title','').strip()
    com = com.replace('@author','').strip()
    com = com.replace('@notice','').strip()
    com = com.replace('@dev','').strip()
    com = com.replace('@param','').strip()
    com = com.replace('#','').strip()
    com = com.replace('@return','return').strip()
    return com

def strip_comment(com):
    com = com.replace('*','').strip()
    com = com.replace('//','').strip()
    com = com.replace('///','').strip()
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
        com = com[1:]
    com = "// " + ' '.join(com.split()[:100]).strip()
    return com

def filter_rows_by_word_count(df, column_name, min_word_count, max_word_count):
    word_count = df[column_name].str.split().apply(len)
    filtered_df = df[(word_count >= min_word_count) & (word_count <= max_word_count)]
    return filtered_df


def remove_comments_from_code(code):
    # Regular expression to match Solidity comments (including nested comments)
    comment_pattern = re.compile(r'\/\/.*|\/\*[\s\S]*?\*\/')
    # Remove comments from the code
    while re.search(comment_pattern, code):
        code = re.sub(comment_pattern, '', code)
    
    code = re.sub(r'\n\s*\n', '\n', code)

    return code

def remove_extra_newlines(code):
    # Use a regular expression to replace consecutive newline characters with a single newline
    result_string = re.sub(r'\n+', '\n', code)
    return result_string


def apply_filter(dataset):
    print("INFO: Filtering Original dataset length:", len(dataset))
    dataset=dataset[dataset['code_string'].apply(lambda x: hasMarker(x))]
    dataset=dataset[dataset['code_string'].apply(lambda x: has_no_license(x))]
    dataset['code_string'] = dataset['code_string'].apply(lambda x: remove_comments_from_code(x))
    dataset['code_string'] = dataset['code_string'].apply(lambda x: remove_extra_newlines(x))
    dataset = dataset.drop_duplicates(subset=['code_string'], keep='first')
    dataset = dataset.drop_duplicates(subset=['comments'], keep='first')
    dataset['comments']=dataset['comments'].apply(lambda x: rm_docstring(x))
    dataset['comments']=dataset['comments'].apply(lambda x: strip_comment(x))
    dataset = filter_rows_by_word_count(dataset, 'code_string', 30, 100)

    print("INFO: Filtering dataset after all filters:", len(dataset))
    return dataset

