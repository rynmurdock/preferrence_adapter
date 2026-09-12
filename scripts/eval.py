
'''
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/eval.py
'''

import os
import torch
torch.set_float32_matmul_precision('high')

from pathlib import Path
from PIL import Image

import webbrowser
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import get_model_and_tokenizer
from config import Config
from data import get_dataloader


def combine_images_side_by_side(images, padding=0, bg_color=(255, 255, 255)):
    images = list(images)
    max_height = max(img.height for img in images)
    total_width = sum(img.width for img in images) + padding * (len(images) + 1)
    combined = Image.new("RGB", (total_width, max_height + padding * 2), bg_color)

    x = padding
    for img in images:
        combined.paste(img, (x, padding))
        x += img.width + padding

    return combined

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


def run_eval(path='/home/ryn_mote/Misc/preferrence-adapter/logs/adapter_only', 
            load_path='/home/ryn_mote/Misc/preferrence-adapter/logs/adapter_only/10500_ckpt'):
    config = Config.from_json(f'{path}/config.json')
    config.load_path = load_path

    # NOTE just faster loading, slower running
    config.do_compile = True
    os.makedirs('scratch/', exist_ok=True, )

    model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                        config.dtype, config.seed, config.do_compile, config)
    model.config.log_dir = './scratch/'
    model.seed = 11

    torch.manual_seed(model.seed)
    __train_dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_path, batch_size=1, num_workers=1, k=config.k,)
    max_samples = 4
    for ind, batch in zip(range(max_samples), val_dataloader):
        model.do_qual_val(batch['sample_pixels'], guidance_scale=5)
        path = f'{model.config.log_dir}/latest_val_{model.seed}_{model.total_steps}.png'
        combine_images_side_by_side(batch['sample_pixels'][0] + [Image.open(path)]).save(f'./scratch/combined_{ind}.png')

#     folder_to_html(model.config.log_dir)

run_eval()


