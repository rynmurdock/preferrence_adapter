import logging
logging.basicConfig(level=logging.INFO)
import os
import json
import torch

from dataclasses import dataclass, field, asdict, fields
from copy import deepcopy


@dataclass
class Config:
    ### Model
    # model_path = None
    transformer_model_path = None
    load_path = None

    seed: int = 13
    # TODO ensure we have even number (pad) so we reach RoPE constraints?
    k: int = 5

    lora_rank: int = 16
    sample_teacher: bool = True

    just_inf_timesteps: bool = True

    # just_inf_timesteps will automatically already shift, 
    #   so this does nothing if just_inf_timesteps=False
    shift_timesteps_resolution: bool = True
    # TODO uniform vs logit normal

    quantize_adam: bool = False
    quantize_model: bool = False

    ### Hparams
    batch_size: int = 8
    lr: float = 1e-4

    # TODO cut to length of content, not 8

    # mainly acts as attention sink, keeping in-domain
    use_prompt: str = 'The scene.'
    # teacher gives the input image back in most cases
    #   sans instruction
    teacher_use_prompt: str = ''

    ### Training
    epochs: int = 3000000000000
    max_steps: int = 100_000
    max_val_steps: int = 64

    # this may break with LoRA teacher switching on/off
    do_compile: bool = True
    device: str = 'cuda:0'
    
    # specifically for *mixed precision*
    # we parse torch dtypes to str on saving & then back on loading for simplicity
    dtype: torch.dtype = field(default=torch.bfloat16, repr=False)
    activation_checkpointing: bool = True

    ### Data
    data_path: str = '../preferrence-set-to-x/PAMELA/annotations/pamela_train.json'
    val_data_path: str = '../preferrence-set-to-x/PAMELA//annotations/pamela_val_unseen.json'
    num_workers: int = 20
    # width & height side lengths
    resolution: tuple[int, int] = (768, 384)


    ### Logging
    exp_name: str = None
    save_path: str = './'
    freq: int = 500 # how often we save/log/etc.

    def to_json(self, filename):
        # we don't want to mutate our actual class
        cfg = deepcopy(self)

        # e.g. torch.bfloat16 -> bfloat16
        cfg.dtype = str(cfg.dtype).split('.')[-1]
        with open(filename, "w") as f:
            json.dump(asdict(cfg), f)

    @classmethod
    def from_json(cls, filename):
        '''
            An unnecessary class method added solely to horrify non-CS juniors. "o.o.p."s
            Loads a config json file using the config class to make a config object.
        '''

        with open(filename, "r") as file:
            data = json.load(file)

            # Filter kwargs to only include valid parameters
            valid_keys = [field.name for field in fields(cls)]
            valid_kwargs = {k: v for k, v in data.items() if k in valid_keys}
            nonviable_kwargs = {k: v for k, v in data.items() if not k in valid_keys}
            if len(nonviable_kwargs) > 0: logging.warning(
                f"{nonviable_kwargs} are not used in our config, so we're dropping them!")
            config = cls(**valid_kwargs)
        config = parse_dtype(config)
        return config

def parse_dtype(config):
    if not isinstance(config.dtype, torch.dtype):
        if isinstance(config.dtype, str):
            try:
                logging.info(f'{config.dtype=}')
                torch_dtype = getattr(torch, config.dtype)
                config.dtype = torch_dtype
            except Exception as e:
                logging.error(f'Error trying to parse dtype: {e}')
                raise(Exception)
        else:
            assert False, f'{config.dtype} is not a torch dtype'
    return config


def verify_config_validity(config):
    parse_dtype(config)

    assert not (config.quantize_model and config.lora_rank), (
            'Saving LoRAs on quantized models is broken, so would need to patch the fn.')

main_config = Config()

if __name__ == "__main__":
    # TODO use pytest insteadf
    orig_main_config = Config()
    orig_main_config.to_json('./placeholder_conf.json')
    new_conf = Config.from_json('./placeholder_conf.json')
    assert orig_main_config == new_conf, (
        'Original and reloaded configs are not equal!'
        f'\n{new_conf=}'
        f'\nOriginal: {orig_main_config=}')
    os.remove('./placeholder_conf.json')


