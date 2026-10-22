'''
Persona judge for generated images: the nearest persona centre in CLIP-L space, or None
if the image is further centre than 95% of the persona's training images. 
'''

import torch
from PIL import Image
from metrics.clip_l import ClipL
from personas import load_personas, image_groups


class PersonaJudge:
    def __init__(self, names, device, root='PAMELA'):
        """
        names: persona names to judge against
        """
        self.names, self.clip, self.root = list(names), ClipL(device), root
        train = set(image_groups(['train']))
        centres, thresholds = [], []
        for members in load_personas(self.names).values():
            X = self.embed([p for p in members if p in train])
            c = X.mean(0)
            c = c / c.norm()
            centres.append(c)
            thresholds.append((X @ c).quantile(0.05))
        self.centres, self.thresholds = torch.stack(centres), torch.stack(thresholds)

    def embed(self, images, batch=32):
        '''PIL images or PAMELA paths -> [n, 768] unit-norm CLIP-L embeddings, in batches'''
        load = lambda x: Image.open(f'{self.root}/{x}').convert('RGB') if isinstance(x, str) else x
        return torch.cat([self.clip([load(x) for x in images[i:i + batch]]) for i in range(0, len(images), batch)])

    @torch.no_grad()
    def __call__(self, images):
        '''PIL images or PAMELA paths -> (persona name or None per image, [n, personas] cosine similarities)'''
        S = self.embed(images) @ self.centres.T
        best = S.argmax(1)
        picks = [self.names[b] if S[i, b] >= self.thresholds[b] else None for i, b in enumerate(best.tolist())]
        return picks, S
