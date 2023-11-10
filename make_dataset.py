#### Create dataset for training
import os
import random
import string
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from solidity_parser.parser import prettify, get_file_content, fragment_code
from utils.parallel import parallelize_on_rows
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count()*2)
tqdm.pandas()

# Only Sol source files and 
contracts_dirs_saved = './slither_processed_contracts.pkl'
sourcify_contracts = pd.read_pickle(contracts_dirs_saved)[:100]

temp = './temp/'

# load starcoder solidity dataset
# starcoder_ds = load_dataset("bigcode/the-stack-dedup", data_dir="data/solidity", split="train")

# load starcoder solidity dataset
# audit_con_ds = load_dataset("mwritescode/slither-audited-smart-contracts", 'all-multilabel')


def generate_random_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(letters_and_digits) for _ in range(length))
    return random_string

def pretiffy_content(content):
    f_name = os.path.join(temp, generate_random_string(25) + '.sol')
    with open(f_name, 'r') as fh:
        fh.write(content)
    
    prettify(f_name)
    return f_name

def clean_comment(comments:list):
    cleaned = []
    for c in comments:
        if c.strip().startswith('///'):
            cleaned.append( c.strip().replace('///', ''))
        elif c.strip().startswith('//'):
            cleaned.append( c.strip().replace('//', ''))
        elif c.strip().startswith('/*'):
            cleaned.append( c.strip().replace('/*', ''))
        elif c.strip().startswith('/**'):
            cleaned.append( c.strip().replace('/**', ''))
        elif c.strip().startswith('*'):
            cleaned.append( c.strip().replace('*', ''))
        elif c.strip().startswith('*/'):
            cleaned.append( c.strip().replace('*/', ''))
        else:
            cleaned.append(c) # unlikely to happen
    return ''.join(cleaned)

def process_file(row):
    result = []
    prettify(row['sol_file'])
    code_and_comment = fragment_code(get_file_content(row['sol_file']))

    for cm, cd in code_and_comment:
        cm = clean_comment(cm)
        result.append({'file_name':row['sol_file'], 'comments':cm, 'code_string':''.join(cd)})

    return pd.DataFrame(result)

# pretify all sol files and retrieve code/comment data
sourcify_contracts['sol_file'] = sourcify_contracts['sol_file'].progress_apply(lambda x: x.replace('/home/pippertetsing/sourcify_contract_data/', './'))
#process_file(sourcify_contracts.iloc[0])

#sourcify_contracts = parallelize_on_rows(sourcify_contracts, process_file, num_of_processes=os.cpu_count())

sourcify_contracts['code_and_comment'] = sourcify_contracts.parallel_apply(lambda x: process_file(x), axis=1)

dataset = pd.DataFrame()
for _, row in sourcify_contracts.iterrows():
    dataset = pd.concat([dataset, row.code_and_comment])

dataset = dataset.reset_index(drop=True)
dataset.to_pickle(f'./test_data.pkl')





