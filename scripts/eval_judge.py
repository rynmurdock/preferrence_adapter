'''
Test the persona judge on held-out core images 
Output: docs/results/S2_judge/.
Run with: python scripts/eval_judge.py [--personas c03_abstract c06_food ...]
'''

import os, argparse, collections
import numpy as np
from personas import select_personas, load_personas, image_groups
from metrics.persona_judge import PersonaJudge

p = argparse.ArgumentParser()
p.add_argument('--personas', nargs='*', default=None, help='persona names; default: all with >= 32 held-out core images')
args = p.parse_args()
OUT = 'docs/results/S2_judge'
os.makedirs(OUT, exist_ok=True)

names = args.personas or select_personas()
judge = PersonaJudge(names, 'cuda')
train = set(image_groups(['train']))
truth, paths = zip(*[(n, p) for n, m in load_personas(names).items() for p in m if p not in train])
picks, S = judge(list(paths))

# accuracy, none rate, and how often the nearest centre alone is right
acc = np.mean([t == q for t, q in zip(truth, picks)])
none = np.mean([q is None for q in picks])
nearest = np.mean([t == names[b] for t, b in zip(truth, S.argmax(1).tolist())])
C = collections.Counter(zip(truth, picks))
with open(f'{OUT}/confusion.tsv', 'w') as f:
    f.write(f'# {len(paths)} held-out core images, {len(names)} personas: accuracy {acc:.3f}, '
            f'none rate {none:.3f}, nearest-centre accuracy {nearest:.3f}\n')
    f.write('true \\ judged\t' + '\t'.join(names + ['None']) + '\n')
    for t in names:
        f.write(t + '\t' + '\t'.join(str(C[t, q]) for q in names + [None]) + '\n')
print(open(f'{OUT}/confusion.tsv').read())

# mistakes sheet: up to 36 random wrongly judged held-out images, captioned "true -> judged"
import random
from PIL import Image, ImageDraw
wrong = [(t, q, p) for t, q, p in zip(truth, picks, paths) if t != q]
wrong = random.Random(0).sample(wrong, min(36, len(wrong)))
T = 224
if wrong:
    img = Image.new('RGB', (T * 6, (T + 14) * ((len(wrong) + 5) // 6)), 'white')
    d = ImageDraw.Draw(img)
    for i, (t, q, p) in enumerate(wrong):
        x, y = (i % 6) * T, (i // 6) * (T + 14)
        img.paste(Image.open(f'PAMELA/{p}').convert('RGB').resize((T, T)), (x, y + 14))
        d.text((x + 2, y + 1), f'{t} -> {q}', fill='black')
    img.save(f'{OUT}/mistakes.jpg', quality=85)
