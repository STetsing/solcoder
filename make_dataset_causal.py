#### Create dataset for training
import os, sys
import random
import string
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets, Dataset, DatasetDict
from utils.get_tokens_causal import * 

tqdm.pandas()
pd.options.mode.copy_on_write = True

# Only Sol source files and 
contracts_dirs_saved = './slither_processed_contracts.pkl'
sourcify_contracts = pd.read_pickle(contracts_dirs_saved)
sourcify_contracts = sourcify_contracts.drop([ 'slither_processed', 'contracts_dirs','has_src_files', 'slither'], axis=1)
print('INFO: length Sourcify solidity dataset:', len(sourcify_contracts))


temp = './temp/'
os.makedirs(temp, exist_ok=True)

# load starcoder solidity dataset
starcoder_df = pd.DataFrame(load_dataset("bigcode/the-stack-dedup", data_dir="data/solidity", split="train", trust_remote_code=True))
starcoder_df["source_code"] = starcoder_df["content"]
starcoder_df = starcoder_df[["source_code", "size"]]
print('INFO: length starcoder solidity dataset:', len(starcoder_df))

# load starcoder solidity dataset
audit_con_df = pd.DataFrame(load_dataset("mwritescode/slither-audited-smart-contracts", 'all-multilabel', split="train", trust_remote_code=True))
audit_con_df = audit_con_df[["source_code"]]
print('INFO: length audited smart contract dataset:', len(audit_con_df))

def read(f):
    with open(f, 'r') as file:
        content = file.read()
    return content

# pretify all sol files and retrieve code/comment data
sourcify_contracts['sol_file'] = sourcify_contracts['sol_file'].progress_apply(lambda x: x.replace('/home/pippertetsing/sourcify_contract_data/', './'))

sourcify_contracts['source_code'] = sourcify_contracts['sol_file'].progress_apply(lambda x: read(x))

def clean_columns(df, keep:list):
    cols_to_remove = df.column_names
    for c in keep:
        cols_to_remove.remove(c)
    df.remove_columns(cols_to_remove)

sourcify_contracts = sourcify_contracts[['source_code']]
starcoder_df = starcoder_df[['source_code']]
audit_con_df = starcoder_df[['source_code']]
sourcify_contracts = sourcify_contracts.drop_duplicates(subset=['source_code'])
starcoder_df = starcoder_df.drop_duplicates(subset=['source_code'])
audit_con_df = audit_con_df.drop_duplicates(subset=['source_code'])


dataset = concatenate_datasets([Dataset.from_pandas(sourcify_contracts), Dataset.from_pandas(starcoder_df)])
dataset = concatenate_datasets([dataset, Dataset.from_pandas(audit_con_df)])

dataset = dataset.train_test_split(test_size=0.2)
test_valid = dataset['test'].train_test_split(test_size=0.5)

split_dataset = DatasetDict({
                            'train': dataset['train'],
                            'test': test_valid['test'],
                            'valid': test_valid['train']
                            })


split_dataset.save_to_disk('./SolCausal')