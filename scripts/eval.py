
'''
python scripts/eval.py
'''

import os
import torch

from pathlib import Path
import webbrowser
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import get_model_and_tokenizer
from config import Config
from data import get_dataloader


def folder_to_html(folder):
        fmt = '.png'

        path = Path(folder).expanduser().resolve()
        if not path.is_dir():
                raise NotADirectoryError(path)

        # %% write HTML file to display all graphs
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <title>{path.name} Plots</title>
        <style>
        img {{
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 100%;
        }}
        </style>
        </head>
        <body>
        """

        for img_file in path.glob(f"*.{fmt}"):
                html += f"""
                <figure>
                <img src="{img_file.name}" alt="{img_file.stem}">
                <figcaption>{img_file.stem}</figcaption>
                </figure>
                """

        html += """
        </body>
        </html>
        """

        html_path = path / "index.html"
        html_path.write_text(html)
        webbrowser.open(html_path.as_uri())


def run_inf(path='/home/ryn_mote/Misc/prior-adapter/logs/hydrovane_Philathea_Kullervo/', 
            load_path='/home/ryn_mote/Misc/prior-adapter/logs/hydrovane_Philathea_Kullervo/3000_ckpt'):
    config = Config.from_json(f'{path}/config.json')
    config.load_path = load_path

    # NOTE just faster loading, slower running
    config.do_compile = False
    os.makedirs('scratch/', exist_ok=True, )

    model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                        config.dtype, config.seed, config.do_compile, config)
    model.config.log_dir = './scratch/'

    model.config.seed = 11
    torch.manual_seed(model.config.seed)
    __train_dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_path, batch_size=1, num_workers=1, k=config.k,)
    for batch in val_dataloader:
        model.do_qual_val(batch['sample_pixels'], guidance_scale=1.2)
        break

    folder_to_html(model.config.log_dir)

run_inf()




