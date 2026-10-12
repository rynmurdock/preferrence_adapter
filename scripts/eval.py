
'''
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/eval.py
'''

import os
import torch
torch.set_float32_matmul_precision('high')

from pathlib import Path
from PIL import Image, ImageFont, ImageDraw

import webbrowser
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import get_model_and_tokenizer
import config as cnf
from data import get_dataloader

def combine_images_side_by_side(images, padding=0, bg_color=(255, 255, 255), scores=None):
    images = list(images)
    if not images:
        return Image.new("RGB", (1, 1), bg_color)

    max_height = max(img.height for img in images)
    total_width = sum(img.width for img in images) + padding * (len(images) + 1)
    combined = Image.new("RGB", (total_width, max_height + padding * 2), bg_color)

    if isinstance(scores, torch.Tensor):
        scores = scores.detach().cpu().tolist()
    scores = list(scores) if scores is not None else []
    scores.extend([None] * (len(images) - len(scores)))

    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", size=40
    )
    draw = ImageDraw.Draw(combined)

    x = padding
    for img, score in zip(images, scores):
        combined.paste(img.convert("RGB"), (x, padding))

        label = f"Score: {score}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        tx, ty = x + 5, padding + 5
        draw.rectangle(
            (tx, ty, tx + text_width + 10, ty + text_height + 10),
            fill=(0, 0, 0),
        )
        draw.text((tx + 5, ty + 5), label, fill=(255, 255, 255), font=font)

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


def run_eval(path=None, 
            load_path=None):

    # TODO must set back to loading from json
    config = cnf.main_config
    if load_path:
        config.load_path = load_path

    # NOTE just faster loading, slower running
    config.do_compile = True
    os.makedirs('scratch/', exist_ok=True, )

    model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                        config.dtype, config.seed, config.do_compile, config)
    model.config.log_dir = './scratch/'
    model.seed = 11

    __train_dataloader, val_dataloader = get_dataloader(config.data_path, config.val_data_path, batch_size=1, num_workers=1, k=config.k,)
    max_samples = 8
    for ind, batch in zip(range(max_samples), val_dataloader):
        torch.manual_seed(model.seed)
        model.do_qual_val(batch['sample_pixels'], guidance_scale=11, 
                          sample_scores=batch['sample_scores'], 
                          n_inference_steps=40, save_images=True)

        path = f'{model.config.log_dir}/None_latest_val_{model.seed}_{model.total_steps}.png'
        combine_images_side_by_side(batch['sample_pixels'][0] + [Image.open(path)], scores=batch['sample_scores'][0]).save(f'./scratch/combined_{ind}_0.png')

        path = f'{model.config.log_dir}/None_latest_val_{model.seed + 1}_{model.total_steps}.png'
        combine_images_side_by_side(batch['sample_pixels'][0] + [Image.open(path)], scores=batch['sample_scores'][0]).save(f'./scratch/combined_{ind}_1.png')


#     folder_to_html(model.config.log_dir)

run_eval()


