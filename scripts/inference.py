
'''
python scripts/inference.py
'''

import os
import torch

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import get_model_and_tokenizer
from config import Config
from data import get_dataloader


def run_inf(path='/home/ryn_mote/Misc/prior-adapter/logs/thermanesthesia_Buckden_Gallicising/', 
            load_path='/home/ryn_mote/Misc/prior-adapter/logs/thermanesthesia_Buckden_Gallicising/18000_ckpt/'):
    config = Config.from_json(f'{path}/config.json')
    config.load_path = load_path
    os.makedirs('scratch/', exist_ok=True, )

    model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                        config.dtype, config.seed, config.do_compile, config)
    model.config.log_dir = './'

    model.config.seed = 11
    torch.manual_seed(model.config.seed)
    __train_dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_path, batch_size=1, num_workers=1, k=config.k,)
    for batch in val_dataloader:
        model.do_qual_val(batch['sample_pixels'], guidance_scale=10)
        break

run_inf()

