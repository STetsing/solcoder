import os
import pandas as pd
from datasets import Dataset, concatenate_datasets, DatasetDict
from utils.get_tokens import * 
from tqdm import tqdm

for i, f in enumerate(os.listdir('./data/')):
    if '.pkl' not in f:
        continue
    #print('INFO: Processing file,', os.path.join('./data/', f))

    if i == 0:
        data = Dataset.from_pandas(pd.read_pickle(os.path.join('./data/', f)))
        data = data.map(process_samples, batched=True, num_proc=None)

        data = data.filter(lambda ex: len(ex['labels'])<=256)
        data = data.remove_columns(['comments', 'context', 'code_string', '__index_level_0__'])
    else: 
        data_nex = Dataset.from_pandas(pd.read_pickle(os.path.join('./data/', f)))
        data_nex = data_nex.map(process_samples, batched=True, num_proc=None)

        data_nex = data_nex.remove_columns(['comments', 'context', 'code_string', '__index_level_0__'])
        data_nex = data_nex.filter(lambda ex: len(ex['labels'])<=256)
        data = concatenate_datasets([data, data_nex])

    if i%60==0 and i !=0:
    
        print(data)
        print('INFO: The dataset size is:', data.data.nbytes / 1e9, 'GB')

        data = data.train_test_split(test_size=0.2)
        test_valid = data['test'].train_test_split(test_size=0.5)

        split_dataset = DatasetDict({
                                    'train': data['train'],
                                    'test': test_valid['test'],
                                    'valid': test_valid['train']
                                    })

        split_dataset.save_to_disk(f'./SolFuncsContext_{str(i)}')
        data = data_nex
