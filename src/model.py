import os
import torch
import logging
from tqdm import tqdm
from copy import deepcopy

from torchvision.transforms import functional as TF

from diffusers import Flux2KleinPipeline
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

from diffusers import BitsAndBytesConfig
from diffusers.training_utils import compute_density_for_timestep_sampling


from transformers import AutoModel, AutoProcessor
import bitsandbytes as bnb
from peft import LoraConfig

from modeling.model_utils import get_inf_timesteps, add_single_stream_embedding_adapter

def ids_encode_pad_mask_images(model, images, dtype):
    with torch.autocast(device_type='cuda', enabled=True, dtype=dtype):
        latents = []
        image_ids = []
        for pil_img in images:
            pil_img = pil_img.resize((pil_img.width//16*16, pil_img.height//16*16))
            img_tensor = TF.to_tensor(pil_img) * 2 - 1  # (3, H, W), values in [-1, 1]
            img_tensor = img_tensor.to(model.device, dtype)[None]
            latent = model.pipe._encode_vae_image(img_tensor, None)
            imids = Flux2KleinPipeline._prepare_image_ids([latent]).to(latent.device)
            image_ids.append(imids[0])
            latents.append(model.pipe._pack_latents(latent)[0])
        padded_latents = torch.nn.utils.rnn.pad_sequence(latents, batch_first=True,).squeeze(1)
        latents_there_mask = torch.nn.utils.rnn.pad_sequence([torch.ones_like(l) for l in latents], 
                                                            batch_first=True, ).squeeze(1) > 0
        image_ids = torch.nn.utils.rnn.pad_sequence(image_ids, batch_first=True).squeeze(1)
        return padded_latents, image_ids, latents_there_mask

def get_loss(model, embeds, images, config, dtype=None,):
    sample_teacher = config.sample_teacher

    dtype = model.dtype if not dtype else dtype
    with torch.no_grad():
        # rng drop out inputs
        zeroing_mask = torch.rand((embeds.shape[0], embeds.shape[1])) < .3
        embeds[zeroing_mask] = 0

        prompt_embeds_attn_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.ones_like(l) for l in embeds.sum(-1)], batch_first=True, ).squeeze(1) > 0
        embeds_ids = model.pipe._prepare_text_ids(embeds).to(model.device)

        x0, typical_image_ids, latents_there_mask = ids_encode_pad_mask_images(model, 
                                                                       images, model.config.dtype)

        noise = torch.randn_like(x0)
        if config.just_inf_timesteps:
            timesteps = get_inf_timesteps(model.pipe.scheduler, x0, num_inference_steps=4, device=model.device)
            k = torch.randint(0, 4, (noise.shape[0],)).to(x0.device)
            timesteps = timesteps[k]
        else:
            u = compute_density_for_timestep_sampling(
                weighting_scheme='uniform',
                batch_size=x0.shape[0],
                logit_mean=0,
                logit_std=1,
            )
            # shift per sample using its mask for seq len of non-padding
            if config.shift_timesteps_resolution:
                mus = []
                for sample_ind in range(noise.shape[0]):
                   mu = compute_empirical_mu(latents_there_mask[sample_ind].amax(-1).sum(0), 4)
                   mus.append(mu)
                mus = torch.tensor(mus).to(u.device, u.dtype)
                u = torch.exp(mus) / (torch.exp(mus) + (1 / u - 1) ** 1)
            indices = (u * model.noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = model.noise_scheduler_copy.timesteps[indices].to(device=x0.device)
        sigma = timesteps.view(-1, 1, 1) / 1000
        latents = sigma * noise + (1 - sigma) * x0

        if sample_teacher:
            if config.lora_rank:
                model.pipe.transformer.orig_transformer.disable_lora()
            
            latent_model_input = torch.cat([latents, x0], dim=1).to(model.pipe.transformer.dtype)
            latent_image_ids = torch.cat([typical_image_ids, typical_image_ids], dim=1)
            teacher_noise_pred = model(latent_model_input, 
                       timesteps=timesteps, image_ids=latent_image_ids, 
                       prompt_embeds=model.pipe.cached_teacher_prompt,
                       txt_ids=model.pipe.cached_teacher_txt_ids,
                       latents_attention_mask=latents_there_mask.repeat(1, 2, 1),
                       prompt_embeds_attn_mask=torch.ones_like(
                           model.pipe.cached_teacher_txt_ids).sum(-1).expand(latents_there_mask.shape[0], -1),
                       vanilla_forward=True,
                       )
            teacher_noise_pred = teacher_noise_pred[:, : latents.size(1) :]
            if config.lora_rank:
                model.pipe.transformer.orig_transformer.enable_lora()


    with torch.autocast(device_type='cuda', enabled=not config.quantize_model, dtype=dtype):
        latent_model_input = torch.cat([latents], dim=1).to(model.pipe.transformer.dtype)
        latent_image_ids = torch.cat([typical_image_ids], dim=1)

        output = model(latent_model_input, 
                       timesteps=timesteps, image_ids=latent_image_ids,
                       prompt_embeds=embeds,
                       txt_ids=embeds_ids,
                       latents_attention_mask=latents_there_mask,
                       prompt_embeds_attn_mask=prompt_embeds_attn_mask,
                       )

    if not sample_teacher:
        target = noise - x0
    else:
        target = teacher_noise_pred

    output = output.to(torch.float32)
    target = target.to(torch.float32)
    loss = (target - output)**2
    # mask anywhere we don't have contents
    loss[~latents_there_mask] = 0
    # mean over batch last
    loss = loss.flatten(1).mean(1).mean()

    logging_dict = {'mse_loss': loss.item(),}
    return loss, logging_dict

class Zoo(torch.nn.Module):
    def __init__(self, pipe, device, dtype, seed=0, config=None) -> None:
        super().__init__()
        self.total_steps = 0

        self.pipe = pipe
        self.seed = seed
        # NOTE: dtype is the mixed dtype; transformer is still in float32
        self.device, self.dtype = device, dtype
        self.config = config
        self.noise_scheduler_copy = deepcopy(pipe.scheduler)

        # load the model and processor
        ckpt = "google/siglip2-base-patch16-384"
        self.semantic_encoder_model = AutoModel.from_pretrained(ckpt).eval().to(self.device)
        self.semantic_encoder_processor = AutoProcessor.from_pretrained(ckpt)

    def enrich_with_preference_embedding(self, embeds, scores):
        # we add a timestep-pos-styled embedding to each semantic embedding
        #   to specify the preference for given images to the model

        # TODO can make improvements here
        embeds = embeds + self.pipe.transformer.score_embedder(scores)
        return embeds
        

    @torch.no_grad()
    def get_semantic_embeds(self, batch_images: list[list]):
        embeds_batch = []
        for images in batch_images:
            inputs = self.semantic_encoder_processor(images=images, 
                                                    return_tensors="pt").to(self.device)
            embeds = self.semantic_encoder_model.get_image_features(**inputs,
                                                                    output_hidden_states=True,
                                                                    ).hidden_states[-3].mean(-2)
            embeds_batch.append(embeds)
        embeds_batch = torch.stack(embeds_batch)
        return embeds_batch

    def forward(self, latents, timesteps, image_ids, 
                prompt_embeds, txt_ids, prompt_embeds_attn_mask=None, 
                latents_attention_mask=None, vanilla_forward=False):
        # latents_attention_mask: zeroes where padded & ones where contents of shape [B, S, D]
        if self.config.quantize_model:
            latents = latents.to(torch.float16)
            timesteps = timesteps.to(torch.float16)
            prompt_embeds = prompt_embeds.to(torch.float16) if prompt_embeds is not None else None

        # account for text first in sequence; sum for just the [B, L], make boolean
        if latents_attention_mask is not None:
            latents_attention_mask = latents_attention_mask.sum(-1)
            attention_mask = torch.concat([prompt_embeds_attn_mask, latents_attention_mask], -1) != 0.

        velocity = self.pipe.transformer(
                hidden_states=latents,  # (B, image_seq_len, C)
                timestep=timesteps / 1000,
                guidance=None,
                encoder_hidden_states=prompt_embeds,
                txt_ids=txt_ids,
                img_ids=image_ids,  # B, image_seq_len, 4
                joint_attention_kwargs={'attention_mask': attention_mask,},
                prompt_embeds_attn_mask=prompt_embeds_attn_mask,
                return_dict=False,
                vanilla_forward=vanilla_forward,
                cached_prompt=self.pipe.cached_prompt,
        )[0]
        return velocity

    @torch.no_grad()
    def inference(self, 
                  embeds, 
                  guidance_scale=5,
                  width_height=None, 
                  generator=None):
        assert embeds.shape[0] == 1, f'Must be batch size of 1. {embeds.shape=}'
        width, height = self.config.resolution if not width_height else width_height[0], width_height[1]
        offload_vae_back_to_cpu = False
        # infer vae device from the all params
        if any([p.device != torch.device('cuda:0') for p in self.pipe.vae.parameters()]):
            offload_vae_back_to_cpu = True
            self.pipe.vae = self.pipe.vae.to('cuda')

        # we may use cfg on our cond image
        self.pipe.config.is_distilled = False
        with torch.autocast('cuda'):
            image = self.pipe(
                # just smuggling for our image ids
                num_inference_steps=4,
                guidance_scale=guidance_scale,
                prompt_embeds=embeds,
                negative_prompt_embeds=torch.zeros_like(embeds),
                height=height,
                width=width,
                generator=generator,
            ).images[0]
        if offload_vae_back_to_cpu:
            self.pipe.vae = self.pipe.vae.to('cpu')
        return image

    @torch.no_grad()
    def do_qual_val(self, pref_history_images, guidance_scale=5, 
                    sample_scores=None, target_scores=None):        
        # we do batch_size=1 evaluation for now
        pref_history_images = pref_history_images[:1]
        semantic_embeds = self.get_semantic_embeds(pref_history_images)
        target_embed = semantic_embeds.new_zeros(len(semantic_embeds), 1, semantic_embeds.shape[-1])
        semantic_embeds = torch.cat([target_embed, semantic_embeds], 1)

        # TODO use sample & target scores
        if not sample_scores:
            logging.warning(f'No scores provided -- giving 5s on all')
            scores = torch.tensor([5,]*semantic_embeds.shape[1])
        else:
            logging.warning(
                'Scores provided but not implemented'
                '             -- giving 5s on all')

        semantic_embeds = self.enrich_with_preference_embedding(semantic_embeds, scores)
        for ind in [self.seed, self.seed+179]:
            width, height = self.config.resolution
            latent_seed_generator = torch.Generator(device="cuda").manual_seed(ind)
            image = self.inference(semantic_embeds, guidance_scale, (width, height), 
                                   latent_seed_generator)
            logging.info(f'Saving at {self.config.log_dir}/latest_val_{ind}_{self.total_steps}.png')
            image.save(f'{self.config.log_dir}/latest_val_{ind}_{self.total_steps}.png')

    def process_inputs(self, batch):
        # target score goes first
        # TODO for varying prompts, we should put the 
        #   prompt after the embeddings to keep 0th instance as target score
        embeds = self.get_semantic_embeds(batch['sample_pixels'])
        # target embed only holds our score embedding for the to-predict image
        target_embed = embeds.new_zeros(len(embeds), 1, embeds.shape[-1])
        embeds = torch.cat([target_embed, embeds], 1)
        scores = torch.cat([batch['target_scores'], batch['sample_scores']], 1)
        embeds = self.enrich_with_preference_embedding(embeds, scores)
        return embeds

    
    @torch.no_grad()
    def val(self, val_dataloader, max_val_steps, dtype):
        logging.info(f'\nRunning validation for max {max_val_steps}\n')
        # fork_rng temporarily isolates changes
        with torch.random.fork_rng():
            # You can change the seed here locally
            torch.manual_seed(self.seed)

            losses = []
            for index, batch in tqdm(enumerate(val_dataloader)):
                if batch is None:
                    continue

                target_images = batch['target_pixels']
                embeds = self.process_inputs(batch)
                loss, loss_logging_dict = get_loss(self, embeds, target_images, 
                                                   config=self.config,)
                losses.append(loss.item())
                if index >= max_val_steps:
                    return sum(losses) / len(losses)
            self.do_qual_val(batch['sample_pixels'])
            return sum(losses) / len(losses)


def get_prompt_embeds_txt_ids(pipe, prompt, device, dtype=torch.float32):
    p, t_ids = pipe.encode_prompt(prompt=prompt, device=device,)
    p, t_ids = p.to(device, dtype), t_ids.to(device, dtype)
    return p, t_ids

def add_lora(transformer, rank, target_modules):
    transformer_lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank, 
        init_lora_weights="gaussian",
        target_modules=target_modules,
        )
    transformer.add_adapter(transformer_lora_config)
    logging.info(f"""trainable params: 
                 {transformer.num_parameters(only_trainable=True)} 
                 || all params: {transformer.num_parameters()}""")

@torch.no_grad()
def get_model_and_tokenizer(path, device, dtype, seed, do_compile, config):
    global Flux2KleinPipeline
    
    transformer = Flux2Transformer2DModel.from_pretrained("black-forest-labs/FLUX.2-klein-4B" if path is None
                                                           else path, # we save without a subdir
                                                           subfolder=None if path else 'transformer',
                                                           quantization_config=BitsAndBytesConfig(load_in_8bit=True,) if config.quantize_model else None,
                                                           strict=False)
    target_modules = [
                "to_q", "to_k", "to_v", "to_out.0",          # double-stream attention
                "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",  # double-stream cross/context attention
                "to_qkv_mlp_proj.0",                            # single-stream fused qkv+mlp-in
                "to_out.0",                                     # single-stream fused attn-out+mlp-out
            ]
    if config.load_path:
        if config.lora_rank:
            transformer.load_lora_adapter(f'{config.load_path}/pytorch_lora_weights.safetensors',
                                      prefix=None,
                                      adapter_name='default',
                                      target_modules=target_modules
                                      )        
            transformer.set_adapters('default', 1)

    elif config.lora_rank:
        # we need a new lora as we aren't loading one
        # inplace operation
        add_lora(transformer, config.lora_rank, target_modules)

    if config.batch_size > 1:
        from modeling.klein_batched_rope import batchify_transformer_rope
        transformer = batchify_transformer_rope(transformer)

    pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B", 
                                              transformer=transformer,
                                              # full precision weights
                                              torch_dtype=torch.float32,
                                              # we'll put things onto cuda ourselves
                                              device='cpu'
                                              ).to('cpu')

    if config.activation_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()

    pipe.vae = pipe.vae.to(device, dtype)
    if do_compile:
        pipe.vae.decode = torch.compile(pipe.vae.decode)
        pipe.vae.encode = torch.compile(pipe.vae.encode)
    assert not any([p.device != torch.device('cuda:0') for p in pipe.vae.parameters()]), [n for n, p in pipe.vae.named_parameters() if p.device != torch.device('cuda:0')]

    pipe.cached_teacher_prompt, pipe.cached_teacher_txt_ids = None, None
    pipe.cached_prompt, pipe.cached_txt_ids = None, None

    if isinstance(config.use_prompt, str):
        logging.info('Caching prompt.')
        pipe.text_encoder = pipe.text_encoder.to(config.device)
        pipe.cached_prompt, pipe.cached_txt_ids = get_prompt_embeds_txt_ids(pipe,
                                                                            config.teacher_use_prompt,
                                                                            config.device,)

    if isinstance(config.teacher_use_prompt, str):
        logging.info('Caching prompt for our teacher.')
        if not pipe.cached_prompt is None:
            pipe.text_encoder = pipe.text_encoder.to(config.device)
        pipe.cached_teacher_prompt, pipe.cached_teacher_txt_ids = get_prompt_embeds_txt_ids(pipe,
                                                                                            config.teacher_use_prompt,
                                                                                            config.device,)
    del pipe.text_encoder
    torch.cuda.empty_cache()

    pipe.transformer = add_single_stream_embedding_adapter(pipe.transformer).to(device)
    pipe.transformer = pipe.transformer.to(device)

    if do_compile:
        pipe.transformer = torch.compile(pipe.transformer)

    pipe.transformer.cached_prompt = pipe.cached_prompt
    pipe.transformer.cached_txt_ids = pipe.cached_txt_ids
    pipe.transformer.k = config.k

    model = Zoo(pipe, config.device, config.dtype, seed, config=config).to(device)
    # we load the LoRA early but apply the adapter in __init__
    if config.load_path:
        adapter_states = torch.load(f'{config.load_path}/adapter.pt')
        transformer.load_state_dict(adapter_states, strict=False)
    return model

def get_optimizer_and_lr_sched(params, lr, config):
    if config.quantize_adam:
        optimizer = bnb.optim.PagedAdamW8bit(params, lr=lr)
    else:
        optimizer = torch.optim.AdamW(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, end_factor=.1, total_iters=100)
    return optimizer, scheduler
