
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



def run_inf(path='/home/ryn_mote/Misc/preferrence-adapter/logs/contd_Romeon_Un-socratic/', 
            load_path='/home/ryn_mote/Misc/preferrence-adapter/logs/contd_Romeon_Un-socratic/92000_ckpt/'):
    config = Config.from_json(f'{path}/config.json')
    config.load_path = load_path

    # TODO clear scratch before using it
    os.makedirs('scratch/', exist_ok=True, )

    model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                        config.dtype, config.seed, config.do_compile, config)
    model.config.log_dir = './scratch/'

    model.config.seed = 11
    torch.manual_seed(model.config.seed)
    image_paths = ['./images/dde6221f-663d-452b-8579-0806cf267df1.png', './images/dde6b3dc-1d07-4647-afb6-cf1c06d5167a_NanoBananaPro.png', './images/de023dbe-e4ae-4f20-8f4d-5df4c801c15e.png', './images/de33f9ac-c607-4fdd-a4f1-2ffe7bcc173c_NanoBananaPro.png', './images/dff6d8e4-4a0c-4e5d-8ff9-1bbdb8389314_NanoBananaPro.png', './images/e0993b4a-ac55-4217-a115-7fb861d6af6a.png', './images/e0d428ed-1b87-471e-ab71-892993b35991.png', './images/e497a118-c6df-4d50-b4db-cd5126346675_NanoBananaPro.png', './images/e5b8b2a2-cd81-43dc-bf8c-a2f712d51745.png', './images/e6346f7d-d316-4f19-8524-810f86602450.png', './images/e6356887-147a-4b34-bfe7-b30397ebe1ab_NanoBananaPro.png', './images/e8a79a9c-46ef-40ed-9383-8063f01cfd04.png', './images/ea20af6c-c410-4d16-9cfb-f6c355c12c43.png', './images/ec19e7f8-d465-40d6-af74-08e23bca41a3.png', './images/ed0daa8b-0ddc-4f12-8a28-1dba84b3f055.png', './images/ed73ffdf-51bf-4381-b3a4-9a244c742a84_NanoBananaPro.png', './images/edcdafe8-02d8-4d8b-b690-d23d7a436cee.png', './images/ee04fa0c-46e2-48f3-b198-6ccb53473de7_NanoBananaPro.png', './images/ee525dc8-8ac5-4067-b02f-59cade3c36e9.png', './images/ee525ffe-d7cd-42e0-a7ff-ee74dfe68c9c.png', './images/efbb3733-73f2-48d3-87c8-910ed2faf6bf.png', './images/f0bbc4d0-720e-49b8-9c77-5798102a9d37_NanoBananaPro.png', './images/f15f31f6-b52e-4fe7-bb19-51b1b55d1fd4.png', './images/f1840085-54f5-4944-ac81-a6c021d3d390.png', './images/f1f15afc-1105-4f31-9232-9e974aed85c6.png', './images/f27fbf20-46a7-4061-9c6a-003735549e02.png', './images/f3ed27ee-6540-463f-8e3f-0935a4cf5d89_NanoBananaPro.png', './images/f50d9462-0021-4a5f-bc83-8428faf9fa5f.png', './images/f5545aa5-cc46-4625-aa02-542b18f79d4d_NanoBananaPro.png', './images/f5b887ef-6d92-4611-b833-d82e1018e03b.png', './images/f61d6b50-e61c-4ea9-ac44-80877bf03916_NanoBananaPro.png', './images/f63dfdf7-34d2-404b-9451-1aa17604ca63.png', './images/f665b018-6556-4ae4-a224-5adf1bf67b08_NanoBananaPro.png', './images/f803bd66-599c-4670-91d9-d3f62a2cfab7.png', './images/f83c3c30-250e-41ff-a8f4-124162383b96.png', './images/f8b952c0-0cad-4b4d-8d0f-6c985424a50d_NanoBananaPro.png', './images/fa4ebca5-d3e8-41a7-913d-5b900ca57c27.png', './images/fb53d201-6204-44d6-8953-0800882b9ee4_NanoBananaPro.png', './images/fd0e5277-659b-4ffb-8f02-d422be222363.png', './images/fdb844aa-f02f-4e82-ab5e-6e168b634241.png', './images/fe2fd0f6-fa79-42d7-8abe-fe311aca461b.png', \
                   './images/ffbb56d2-e0d9-478c-b9a5-d8d049825663_NanoBananaPro.png'][:32]
    images = [Image.open(f'PAMELA/{i}').convert('RGB') for i in image_paths]
    model.do_qual_val(images, guidance_scale=14, n_inference_steps=40, sample_scores=[5]*len(image_paths),
                      save_images=True)

run_inf()




