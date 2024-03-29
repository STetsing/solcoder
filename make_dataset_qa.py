#### Create dataset for training
import os, sys
import random
import string
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from solidity_parser.updated_parser import get_comments_code_blocks
from utils.parallel import parallelize_on_rows
from utils.data_filtering import apply_filter
from pandarallel import pandarallel

enable_pretty = False

pandarallel.initialize(progress_bar=True, nb_workers=15)
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
starcoder_df = pd.DataFrame(load_dataset("bigcode/the-stack-dedup", data_dir="data/solidity", split="train"))
starcoder_df["source_code"] = starcoder_df["content"]
starcoder_df = starcoder_df[["source_code", "size"]]
print('INFO: length starcoder solidity dataset:', len(starcoder_df))

# load starcoder solidity dataset
audit_con_df = pd.DataFrame(load_dataset("mwritescode/slither-audited-smart-contracts", 'all-multilabel', split="train"))
audit_con_df = audit_con_df[["source_code"]]
print('INFO: length audited smart contract dataset:', len(audit_con_df))


def generate_random_string(length):
    letters_and_digits = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(letters_and_digits) for _ in range(length))
    return random_string

def process_content(row, pretty=True if enable_pretty else False):
    try:
        f_name = os.path.join(temp, generate_random_string(25) + '.sol')
        with open(f_name, 'w') as fh:
            fh.write(row['source_code'])
            fh.close()
        
        if pretty:
            prettify(f_name)

        result = []
        code_and_comment = get_comments_code_blocks(f_name)
        for cm, cd in code_and_comment:
            # result.append({'context':ctx, 'comments':cm, 'code_string':''.join(cd)})
            result.append({'comments':cm, 'code_string':''.join(cd)})

        return pd.DataFrame(result)
    except Exception as ex:
        raise ValueError("WARNING: error occured while processing content", ex)
        return pd.DataFrame()
    finally:
        if os.path.exists(f_name):
            os.remove(f_name)

def process_file(row):
    result = []
    
    if enable_pretty:
        prettify(row['sol_file'])

    code_and_comment = get_comments_code_blocks(row['sol_file'])

    for cm, cd in code_and_comment:
        # result.append({'context':ctx, 'comments':cm, 'code_string':''.join(cd)})
        result.append({'comments':cm, 'code_string':''.join(cd)})

    return pd.DataFrame(result)

# pretify all sol files and retrieve code/comment data
sourcify_contracts['sol_file'] = sourcify_contracts['sol_file'].progress_apply(lambda x: x.replace('/home/pippertetsing/sourcify_contract_data/', './'))

#sourcify_contracts['code_and_comment'] = sourcify_contracts.progress_apply(lambda x: process_file(x), axis=1)

# print('\nINFO: Processing audited contract data')
# audit_con_df['code_and_comment'] = audit_con_df.parallel_apply(lambda x: process_content(x), axis=1)

# print('\nINFO: Processing starcoder data')
# starcoder_df['code_and_comment'] = starcoder_df.parallel_apply(lambda x: process_content(x), axis=1)


print('\nINFO: appending sourcify results')
step = 10000
for i in range(0, len(sourcify_contracts), step):
    print('\nINFO: processing sourcify batch', i)
    small_set = sourcify_contracts[i: i+step]
    small_set['code_and_comment'] = small_set.parallel_apply(lambda x: process_file(x), axis=1)

    dataset = pd.DataFrame()
    for _, row in small_set.iterrows():
        dataset = pd.concat([dataset, row.code_and_comment])

    print('Info: ourcify Size of dataset batch', i, len(dataset))
    dataset = apply_filter(dataset)
    dataset.to_csv(f'./data/sourcify_{i}_comment_code_sol.csv')
    dataset = pd.DataFrame()


print('\nINFO: appending audited contracts results')
for i in range(0, len(audit_con_df), step):
    print('\nINFO: processing audited contract batch', i)
    small_set = audit_con_df[i: i+step]
    small_set['code_and_comment'] = small_set.parallel_apply(lambda x: process_content(x), axis=1)

    dataset = pd.DataFrame()
    for _, row in small_set.iterrows():
        dataset = pd.concat([dataset, row.code_and_comment])
    
    print('Info: audited Size of dataset batch', i, len(dataset))
    dataset = apply_filter(dataset)
    dataset.to_csv(f'./data/audited_{i}_comment_code_sol.csv')
    dataset = pd.DataFrame()


print('\nINFO: appending starcoder contracts results')
for i in range(0, len(starcoder_df), step):
    print('\nINFO: processing starcoder_df batch', i)
    small_set = starcoder_df[i: i+step]
    small_set['code_and_comment'] = small_set.parallel_apply(lambda x: process_content(x), axis=1)

    dataset = pd.DataFrame()
    for _, row in small_set.iterrows():
        dataset = pd.concat([dataset, row.code_and_comment])
    
    print('Info: starcoder Size of dataset batch', i, len(dataset))
    dataset = apply_filter(dataset)
    dataset.to_csv(f'./data/starcoder_{i}_comment_code_sol.csv')
    dataset = pd.DataFrame()

print('\nINFO: writing dataset to disk')
bigset = pd.DataFrame()






