'''
CLIP ViT-L/14 image embeddings, for building the personas and judging generated images.
CLIP-L instead of SigLIP, so the judge does not share the adapter's encoder.
'''
import torch
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor

CLIP_L = 'openai/clip-vit-large-patch14'


class ClipL(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = CLIPVisionModelWithProjection.from_pretrained(CLIP_L)
        self.processor = CLIPImageProcessor.from_pretrained(CLIP_L)
        self.device = device
        self.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        '''list of PIL images -> [n, 768] unit-norm embeddings, so a dot product is a cosine similarity'''
        e = self.model(**self.processor(images=images, return_tensors='pt').to(self.device)).image_embeds
        return e / e.norm(dim=-1, keepdim=True)