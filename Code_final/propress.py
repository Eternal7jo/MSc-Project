#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import tqdm
import os

for root, dirs, files in tqdm(os.walk('data/python')):
    for name in files:
        if name.endswith('.jsonl'):
            print(f'Joining {os.path.join(root, name)}')
            combined_name = name.split('_')
            combined_name = f'{combined_name[0]}_{combined_name[1]}.jsonl'
            with open(os.path.join(root, name), 'r') as f:
                contents = f.read()
            with open(os.path.join(root, combined_name), 'a+') as f:
                f.write(contents)
                f.write('\n')
            os.system(f'rm {os.path.join(root, name)}')