'''
Image-quality scores for validation images.

For now, we'll just use the LAION aesthetic predictor v2 (https://github.com/christophschuhmann/improved-aesthetic-predictor).
HPSv2 uses prompt-based aesthetic scoring, but we don't have a prompt for each validation image yet.

'''
import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor


class AestheticScorer(torch.nn.Module):
    '''
    LAION aesthetic predictor v2: an MLP (Linear)on the normalized CLIP ViT-L/14 image embedding (1-10 scale).

    References:
      - Original weights (LAION): https://github.com/christophschuhmann/improved-aesthetic-predictor
        file sac+logos+ava1-l14-linearMSE.pth
      - HF port we load (shunk031): https://huggingface.co/shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE
        modeling_v2.py, class AestheticsPredictorV2Linear (head, normalization, OpenAI CLIP-L),
        also https://github.com/shunk031/simple-aesthetics-predictor
    '''
    REPO = 'shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE'

    def __init__(self, device):
        super().__init__()
        path = snapshot_download(self.REPO)
        # the predictor's CLIP-L is OpenAI's, unchanged: modeling_v2.py, convert_v2_linear_from_openai_clip()
        self.clip = CLIPVisionModelWithProjection.from_pretrained('openai/clip-vit-large-patch14')
        self.processor = CLIPImageProcessor.from_pretrained(path)
        # layers 1, 3, 5 are dropouts in the original: modeling_v2.py, AestheticsPredictorV2Linear.__init__
        self.head = torch.nn.Sequential(torch.nn.Linear(768, 1024), torch.nn.Identity(),
                                        torch.nn.Linear(1024, 128), torch.nn.Identity(),
                                        torch.nn.Linear(128, 64), torch.nn.Identity(),
                                        torch.nn.Linear(64, 16), torch.nn.Linear(16, 1))
        self.head.load_state_dict({k.removeprefix('layers.'): v for k, v in
                                   load_file(f'{path}/model.safetensors').items() if k.startswith('layers.')})
        self.device = device
        self.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        '''images: list of PIL images -> list of floats'''
        # unit-norm embedding, then the head: modeling_v2.py, AestheticsPredictorV2Linear.forward
        embeds = self.clip(**self.processor(images=images, return_tensors='pt').to(self.device)).image_embeds
        return self.head(embeds / embeds.norm(dim=-1, keepdim=True)).squeeze(-1).tolist()
