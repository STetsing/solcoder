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

enable_pretty = False

pandarallel.initialize(progress_bar=True, nb_workers=os.cpu_count())
tqdm.pandas()

# Only Sol source files and 
contracts_dirs_saved = './slither_processed_contracts.pkl'
sourcify_contracts = pd.read_pickle(contracts_dirs_saved)
print('INFO: length Sourcify solidity dataset:', len(sourcify_contracts))

temp = './temp/'

# load starcoder solidity dataset
starcoder_df = pd.DataFrame(load_dataset("bigcode/the-stack-dedup", data_dir="data/solidity", split="train")[:-1])
starcoder_df["source_code"] = starcoder_df["content"]
print('INFO: length starcoder solidity dataset:', len(starcoder_df))

# load starcoder solidity dataset
audit_con_df = pd.DataFrame(load_dataset("mwritescode/slither-audited-smart-contracts", 'all-multilabel', split="train")[:-1])
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
        code_and_comment = fragment_code(get_file_content(f_name))

        for cm, cd in code_and_comment:
            cm = clean_comment(cm)
            result.append({'file_name':row['source_code'], 'comments':cm, 'code_string':''.join(cd)})

        return pd.DataFrame(result)
    except Exception as ex:
        print("WARNING: error occured while processing content", ex)
        return pd.DataFrame()
    finally:
        if os.path.exists(f_name):
            os.remove(f_name)

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
    
    if enable_pretty:
        prettify(row['sol_file'])

    code_and_comment = fragment_code(get_file_content(row['sol_file']))

    for cm, cd in code_and_comment:
        cm = clean_comment(cm)
        result.append({'file_name':row['sol_file'], 'comments':cm, 'code_string':''.join(cd)})

    return pd.DataFrame(result)

# pretify all sol files and retrieve code/comment data
sourcify_contracts['sol_file'] = sourcify_contracts['sol_file'].progress_apply(lambda x: x.replace('/home/pippertetsing/sourcify_contract_data/', './'))

print('\nINFO: Processing sourcify data')
sourcify_contracts['code_and_comment'] = sourcify_contracts.parallel_apply(lambda x: process_file(x), axis=1)

print('\nINFO: Processing audited contract data')
audit_con_df['code_and_comment'] = audit_con_df.parallel_apply(lambda x: process_content(x), axis=1)

print('\nINFO: Processing starcoder data')
starcoder_df['code_and_comment'] = starcoder_df.parallel_apply(lambda x: process_content(x), axis=1)


dataset = pd.DataFrame()

print('\nINFO: appending sourcify results')
for i in tqdm(range(len(sourcify_contracts))):
    row = sourcify_contracts.iloc[i]
    dataset = pd.concat([dataset, row.code_and_comment])

    if i%10000==0 and i!=0:
        dataset.to_pickle(f'./data/sourcify_{i}_comment_code_sol.pkl')
        dataset = pd.DataFrame()

dataset.to_pickle(f'./data/sourcify_last_comment_code_sol.pkl')
dataset = pd.DataFrame()


print('\nINFO: appending audited contracts results')
for i in tqdm(range(len(audit_con_df))):
    row = audit_con_df.iloc[i]
    dataset = pd.concat([dataset, row.code_and_comment])

    if i%10000==0 and i!=0:
        dataset.to_pickle(f'./data/audited_{i}_comment_code_sol.pkl')
        dataset = pd.DataFrame()

dataset.to_pickle(f'./data/audited_last_comment_code_sol.pkl')
dataset = pd.DataFrame()


print('\nINFO: appending starcoder contracts results')
for i in tqdm(range(len(starcoder_df))):
    row = starcoder_df.iloc[i]
    dataset = pd.concat([dataset, row.code_and_comment])

    if i%10000==0 and i!=0:
        dataset.to_pickle(f'./data/starcoder_{i}_comment_code_sol.pkl')
        dataset = pd.DataFrame()

dataset.to_pickle(f'./data/starcoder_last_comment_code_sol.pkl')
dataset = pd.DataFrame()

print('\nINFO: writing dataset to disk')
bigset = pd.DataFrame()

for f in os.listdir('./data/'):
    if '.pkl' not in f:
        continue
    df = pd.read_pickle(os.path.join('./data/', f))
    bigset = pd.concat([bigset, df])

print('\nINFO: writing dataset to disk')
bigset = bigset.reset_index(drop=True)
bigset.to_pickle(f'./comment_code_sol.pkl')





