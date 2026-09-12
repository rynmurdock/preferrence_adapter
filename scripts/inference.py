
'''
python scripts/inference.py
'''

import os
import torch

from pathlib import Path
from PIL import Image

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


def run_inf(path='/home/ryn_mote/Misc/preferrence-adapter/logs/adapter_only', 
            load_path='/home/ryn_mote/Misc/preferrence-adapter/logs/adapter_only/10500_ckpt'):
    config = Config.from_json(f'{path}/config.json')
    config.load_path = load_path

    # TODO clear scratch before using it
    os.makedirs('scratch/', exist_ok=True, )

    model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                        config.dtype, config.seed, config.do_compile, config)
    model.config.log_dir = './scratch/'

    model.config.seed = 11
    torch.manual_seed(model.config.seed)
    image_paths = ['/home/ryn_mote/Misc/bigDiffusion/assets/3o.png',
                   '/home/ryn_mote/Misc/bigDiffusion/assets/2o.png',
                   '/home/ryn_mote/Misc/bigDiffusion/assets/10o.png',
                   '/home/ryn_mote/Misc/bigDiffusion/assets/9o.png',
                   ]
    images = [Image.open(i).convert('RGB') for i in image_paths]
    model.do_qual_val(images, guidance_scale=9, 
                      sample_scores=[4, 1, 1, 1], target_scores=[5])
    folder_to_html(model.config.log_dir)

run_inf()




