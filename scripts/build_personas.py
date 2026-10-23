'''
Build the simulated personas into docs/results/S1_personas/:
1. embed every PAMELA image with CLIP-L,
2. Louvain communities of the 15-nearest-neighbour cosine graph; an image is "core" if most of its
   neighbours are in its own community,
3. every community becomes a persona whose members are its core images,
'''

import os, json, collections
import numpy as np, torch, networkx as nx
from PIL import Image
from src.personas import PERSONAS_FILE, image_groups
from src.metrics.clip_l import ClipL

OUT = os.path.dirname(PERSONAS_FILE)
os.makedirs(OUT, exist_ok=True)

# 1. CLIP-L embeddings of all PAMELA images assumed cached
groups = image_groups()
paths = sorted(groups)
cache = f'{OUT}/clipL_embeds.pt'
if os.path.exists(cache) and torch.load(cache)['paths'] == paths:
    X = torch.load(cache)['X']
else:
    clip = ClipL('cuda')
    X = torch.cat([clip([Image.open(f'PAMELA/{p}').convert('RGB') for p in paths[i:i + 64]]).cpu()
                   for i in range(0, len(paths), 64)]).float()
    torch.save({'paths': paths, 'X': X}, cache)
    del clip
X = X.numpy()

# 2. Louvain communities of the K-nearest-neighbour cosine graph
K, SEED, CORE = 15, 0, 0.8
S = X @ X.T                                  # cosine similarity between every pair of images
np.fill_diagonal(S, -1)                      # an image is not its own neighbour
nn = np.argsort(-S, 1)[:, :K]                # each image's K most similar images
G = nx.Graph()
G.add_weighted_edges_from((i, int(j), float(S[i, j])) for i in range(len(X)) for j in nn[i])
lab = np.zeros(len(X), int)
for c, m in enumerate(sorted(nx.community.louvain_communities(G, seed=SEED), key=len, reverse=True)):
    lab[list(m)] = c                         # community 0 is the largest
core = (lab[nn] == lab[:, None]).mean(1) >= CORE


# 3. Every community is a persona
personas = {}
for c in range(lab.max() + 1):
    idx = np.where((lab == c) & core)[0]
    top = collections.Counter(groups[paths[i]] for i in idx).most_common(3)
    name = f'c{c:02d}_' + (top[0][0].replace(' ', '_') if top else 'none')
    personas[name] = {'size': int((lab == c).sum()), 'core': len(idx), 'top_labels': top,
                      'members': [paths[i] for i in idx]}
json.dump({'method': f'louvain seed={SEED} on {K}-NN CLIP-L cosine graph, core >= {CORE}',
           'personas': personas}, open(PERSONAS_FILE, 'w'), indent=1)
for name, p in personas.items():
    print(f"{name:28s} size {p['size']:4d}  core {p['core']:4d}  {p['top_labels']}")


# 4. (a) one row per persona: 8 random core images captioned with its size and top labels
from PIL import ImageDraw
rng = np.random.default_rng(0)
T = 128
def sheet(rows, file):
    '''rows: [(caption, [image paths])] -> one row of T-pixel thumbnails per caption, saved as a jpg'''
    img = Image.new('RGB', (T * 8, (T + 16) * len(rows)), 'white')
    d = ImageDraw.Draw(img)
    for r, (caption, ps) in enumerate(rows):
        d.text((2, r * (T + 16) + 2), caption, fill='black')
        for j, p in enumerate(ps[:8]):
            img.paste(Image.open(f'PAMELA/{p}').convert('RGB').resize((T, T)), (j * T, r * (T + 16) + 16))
    img.save(f'{OUT}/{file}', quality=85)
sheet([(f"{n}  size {p['size']}  core {p['core']}  {p['top_labels']}",
        list(rng.choice(p['members'], min(8, p['core']), replace=False))) for n, p in personas.items()],
      'personas.jpg')
# (b) PAMELA's labels against the personas to see how aligned they are.
names = list(personas)
label = np.array([groups[p] for p in paths])
with open(f'{OUT}/labels_vs_personas.tsv', 'w') as f:
    f.write('label\t' + '\t'.join(names) + '\n')
    for l in sorted(set(label)):
        f.write(l + '\t' + '\t'.join(str(int(((lab == c) & (label == l)).sum())) for c in range(len(names))) + '\n')
rare = []
for c, n in enumerate(names):
    for l, cnt in collections.Counter(label[(lab == c) & core]).items():
        if cnt / personas[n]['core'] < 0.05:
            rare.append((cnt, n, l, [paths[i] for i in np.where((lab == c) & core & (label == l))[0]]))
rare.sort(key=lambda r: -r[0])
sheet([(f'core of {n}, but labelled "{l}" ({cnt} images)', list(rng.permutation(ps))) for cnt, n, l, ps in rare[:16]],
      'label_errors.jpg')
