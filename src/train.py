

###########################################
'''
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python src/train.py
'''
###########################################


import os
import torch
torch.set_float32_matmul_precision('high')
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data import get_dataloader, get_demographic_vocab
from config import main_config, verify_config_validity
from model import get_model_and_tokenizer, get_optimizer_and_lr_sched, get_loss
from utils.exp_logging import setup_log_dir

logging.basicConfig(level=logging.INFO)

def main(config):
    verify_config_validity(config)
    if config.load_path or config.transformer_model_path:
        # iterate data's seed when we're loading a checkpoint;
        #   ideally would start at the stopping point, but just new random is better than nothing
        logging.warning('''Advancing training config's seed by 1! for loading a ckpt for training''')
        config.seed = config.seed + 1
        torch.manual_seed(config.seed)

    log_dir = setup_log_dir(config)
    config.log_dir = log_dir
    logging.info(f'''
****************************
Setting up our logging directory at {log_dir} !
****************************
                 ''')
    logging.basicConfig(level=logging.INFO,
                        format=f"[ {log_dir} ] %(message)s",
                        force=True)
    writer = SummaryWriter(log_dir)


    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    model = get_model_and_tokenizer(config.transformer_model_path, config.device, 
                                    config.dtype, config.seed, config.do_compile, config)

    # exclude
    pattern = 'orig_transformer'
    trained_params = dict([(n, p) for n, p in 
                           model.pipe.transformer.named_parameters() if pattern not in n])
    not_trained = [p for n, p in model.pipe.transformer.named_parameters() 
                   if n not in trained_params.keys()]

    if config.lora_rank:
        # include
        pattern = 'lora'
        trained_params.update(dict([(n, p) for n, p in model.pipe.transformer.named_parameters() 
                                    if pattern in n]))
        not_trained = [p for n, p in model.pipe.transformer.named_parameters() 
                           if n not in trained_params.keys()]
    logging.info(f'''
****************************
Training {len(trained_params)} torch modules
****************************
    ''')
    optimizer, lr_sched = get_optimizer_and_lr_sched(list(trained_params.values()), config.lr, config)
    for p in not_trained:
        p.requires_grad = False

    demographic_vocab = get_demographic_vocab(config.data_path)
    dataloader, val_dataloader = get_dataloader(
                config.data_path, 
                config.val_data_path,
                config.batch_size, 
                config.num_workers, 
                k=config.k,
                demographic_vocab=demographic_vocab
            )
    
    train_losses = []
    inner_train_losses = []
    validation_losses = []
    total_inds = 0

    for epoch in range(config.epochs):
        for ind, batch in tqdm(enumerate(iter(dataloader))):
            if total_inds > config.max_steps:
                logging.info('Saving our transformer & ending training')
                if config.lora_rank:
                    model.pipe.transformer.orig_transformer.save_lora_adapter(f'{config.log_dir}/last_epoch_ckpt/',
                                                             adapter_name='default',)
                # Remove `else` branch saving the full Klein weights already on disk
                writer.close()
                return
            if batch is None or \
                            (False and batch.get('latents', None) is None):
                logging.warning(f'Skipping batch! {batch}')
                continue

            images = batch['target_pixels']
            try:
                embeds = model.process_inputs(batch)
            except Exception as e:
                logging.warning(f'{e} when getting semantic embeddings')

            if total_inds % config.freq == 0:
                # NOTE autocasting because our fp32 training model is also our val model
                val_loss, qual_images = model.val(val_dataloader, config.max_val_steps, config.dtype)
                logging.info(f'{val_loss=:.4f}')
                writer.add_scalar('loss/val', val_loss, total_inds)
                for i, qual_image in enumerate(qual_images):
                    writer.add_image(f'qual/sample_{i}', np.array(qual_image), total_inds, dataformats='HWC')
                if torch.cuda.is_available():
                    writer.add_scalar('sys/gpu_mem_allocated_gb', torch.cuda.memory_allocated() / 2**30, total_inds)
                    writer.add_scalar('sys/gpu_mem_reserved_gb', torch.cuda.max_memory_reserved() / 2**30, total_inds)
                if total_inds // config.freq != 0:
                    validation_losses.append(val_loss)
                if len(inner_train_losses) > 0:
                    if total_inds // config.freq != 0:
                        avg_train_loss = sum(inner_train_losses)/len(inner_train_losses)
                        train_losses.append(avg_train_loss)
                        writer.add_scalar('loss/train', avg_train_loss, total_inds)
                    inner_train_losses = []

                plt.plot(train_losses, label='Train')
                plt.plot(validation_losses, label='Validation')
                plt.legend()
                plt.savefig(log_dir + f'/{total_inds}_loss_curve.jpg')
                plt.clf()

                if total_inds == 2 * config.freq:
                    # to save our scaling, we remove the first 2 metric points after they're done
                    train_losses = []
                    validation_losses = []


            loss, loss_logging_dict = get_loss(model, embeds, images, config=config,)
            inner_train_losses.append(loss.item())
            loss.backward()
            # max_norm=inf so this only measures the gradient norm, never clips
            grad_norm = torch.nn.utils.clip_grad_norm_(trained_params.values(), max_norm=float('inf'))
            optimizer.step()
            lr_sched.step()
            optimizer.zero_grad()
            model.total_steps += 1

            writer.add_scalar('loss/train_step', loss.item(), total_inds)
            writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], total_inds)
            writer.add_scalar('train/grad_norm', grad_norm.item(), total_inds)

            total_inds += 1
            if total_inds % config.freq == 0:
                logging.info('Saving our transformer')
                save_dir = f'{config.log_dir}/{total_inds}_ckpt/'
                os.makedirs(save_dir, exist_ok=True)
                if config.lora_rank:
                    # from_pt=True can't be used here
                    model.pipe.transformer.orig_transformer.save_lora_adapter(save_dir,
                                                             adapter_name='default')
                # Remove `else` branch saving the full Klein pretrained weights already on disk
                state_dict = model.pipe.transformer.state_dict()
                # Filter out keys that contain "specific_layer_name" or "unwanted_param"
                filtered_state_dict = {
                    k: v for k, v in state_dict.items() 
                    if "orig_transformer" not in k
                }
                # Save the filtered state dictionary
                torch.save(filtered_state_dict, f"{save_dir}adapter.pt")

if __name__ == '__main__':
    main(main_config)

