import os
import pandas as pd
from datasets import Dataset
from utils.get_tokens import * 
from tqdm import tqdm

for i, f in tqdm(enumerate(os.listdir('./sets/'))):
    if '.pkl' not in f:
        continue
    print('INFO: Processing file,', os.path.join('./sets/', f))
    data = Dataset.from_pandas(pd.read_pickle(os.path.join('./sets/', f))[:100000])
    data = data.map(process_samples, batched=True, batch_size=16, num_proc=9)

    data = data.remove_columns(['comments', 'context', 'code_string', '__index_level_0__'])
    print(data)
    print('INFO: The dataset size is:', data.data.nbytes / 1e9, 'GB')


    break