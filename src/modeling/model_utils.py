import numpy as np
import torch
import inspect

from diffusers.models.transformers.transformer_flux2 import Flux2SingleTransformerBlock


class ScoreEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        time_embed_dim: int = 768,
        out_dim: int = 768,
        post_act_fn: str | None = None,
        cond_proj_dim=None,
        sample_proj_bias=True,
    ):
        super().__init__()

        self.linear_1 = torch.nn.Linear(in_channels, time_embed_dim, sample_proj_bias)

        if cond_proj_dim is not None:
            self.cond_proj = torch.nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None

        self.act = torch.nn.SiLU()

        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = torch.nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)

        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = torch.nn.SiLU()

        # we use one-hot to make our own linear,
        #   as nn.Embedding is apt to break on distributed training
        self.embed_linear = torch.nn.Linear(5, out_dim)

    # TODO infer device, not default to cuda
    def forward(self, score, device='cuda', condition=None):
        # TODO don't snap to int
        # score-1 so we are 0-indexed.
        score_one_hot = torch.nn.functional.one_hot(torch.Tensor(score-1).to(torch.long), 
                                                    num_classes=5).to(device, torch.float)
        sample = self.embed_linear(score_one_hot)
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)

        if self.act is not None:
            sample = self.act(sample)

        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample


class SemanticFlux2SingleTransformerBlock(Flux2SingleTransformerBlock):
    def __init__(
        self, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        joint_attention_kwargs: dict[str,] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        norm_hidden_states = self.norm(hidden_states)
        joint_attention_kwargs = joint_attention_kwargs or {}
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=rotary_emb,
            **joint_attention_kwargs,
        )
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        hidden_states = hidden_states + attn_output
        return hidden_states


class SemanticEmbedsKlein(torch.nn.Module):
    def __init__(self, transformer):
        super().__init__()
        self.cached_prompt = None

        self.dtype = transformer.dtype
        self.device = transformer.device
        self.orig_transformer = transformer
        out_dim = 7680
        # TODO take in siglip config's embed size through its config
        self.adapter = torch.nn.ModuleList([
                SemanticFlux2SingleTransformerBlock(
                    dim=128,
                    num_attention_heads=4,
                    attention_head_dim=128,
                    mlp_ratio=2,
                    eps=1e-6,
                    bias=False,
                )
            for _ in range(3)])
        # following i1 (https://arxiv.org/abs/2606.11289)
        #   for reasonable depth+1

        self.in_linear = torch.nn.Linear(768, 128)
        self.out_linear = torch.nn.Linear(128, out_dim)       
        self.score_embedder = ScoreEmbedding()


    def forward(self, *args, **kwargs):
        # default to using our adapter
        vanilla_forward = True if kwargs.get('vanilla_forward', False) else False
        prompt_embeds_attn_mask = kwargs.get('prompt_embeds_attn_mask')
        cached_prompt = self.cached_prompt

        if not vanilla_forward:
            # single stream over enc hidden states (semantic embeddings)
            txt_ids = kwargs['txt_ids']
            sem_emb_rotary_embeds = self.orig_transformer.pos_embed(txt_ids)
            hidden_states = self.in_linear(kwargs['encoder_hidden_states'])
            for a in self.adapter:
                hidden_states = a(
                    hidden_states=hidden_states,
                    rotary_emb=sem_emb_rotary_embeds, # TODO just use max truncate here?
                    # TODO cut to length of content, not 8
                    joint_attention_kwargs={'attention_mask': prompt_embeds_attn_mask},
                    )
            hidden_states = self.out_linear(hidden_states)
            if not cached_prompt is None:
                hidden_states = torch.cat([cached_prompt.expand(len(hidden_states), -1, -1)[:, :8], 
                    hidden_states], 1)
                kwargs['txt_ids'] = self.cached_txt_ids[:, :8+self.k+1]
                if not kwargs['joint_attention_kwargs'] is None:
                    att_mask = kwargs['joint_attention_kwargs']['attention_mask']
                    att_mask = torch.cat([torch.ones_like(att_mask)[:, :8] > 0, 
                        att_mask], 1)
                    kwargs['joint_attention_kwargs']['attention_mask'] = att_mask
            kwargs['encoder_hidden_states'] = hidden_states

        if 'vanilla_forward' in kwargs: kwargs.pop('vanilla_forward')
        if 'prompt_embeds_attn_mask' in kwargs: kwargs.pop('prompt_embeds_attn_mask')
        if 'cached_prompt' in kwargs: kwargs.pop('cached_prompt')

        out = self.orig_transformer(*args, **kwargs)
        return out

def add_single_stream_embedding_adapter(transformer):
    new_transformer = SemanticEmbedsKlein(transformer)

    new_transformer.config = transformer.config
    new_transformer.cache_context = transformer.cache_context
    return new_transformer


# Copied from diffusers.pipelines.flux2.pipeline_flux2.compute_empirical_mu
def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        mu = a2 * image_seq_len + b2
        return float(mu)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



def get_inf_timesteps(scheduler, latents, num_inference_steps, device, sigmas=None):
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    if hasattr(scheduler.config, "use_flow_sigmas") and scheduler.config.use_flow_sigmas:
        sigmas = None
    image_seq_len = latents.shape[1]
    mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    timesteps, num_inference_steps = retrieve_timesteps(
        scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    return timesteps
