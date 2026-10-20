'''
Validation suite for a preference-adapter checkpoint, on simulated personas.
T0  the checkpoint is fully and exactly in the model.
T1  image swap: each persona's own 32 held-out images, all rated 5.
T2  score swap: one mixed history of the personas' images and each persona rates its own 5, the rest 1.
T3  score flip: T1's histories, all rated 1.

-Each history is generated at seeds 0-7 with the repo's sampler. 
-The persona judge labels each image. 
-Hit rate and an exact permutation p (personas shuffled within a seed) for T1-T3.
-A seed-pair yardstick for image change.

Run with: python scripts/validate.py --load_path checkpoints/ryan_oct14_step39200
'''
import argparse, os, random
import numpy as np
import torch
from safetensors.torch import load_file
from config import main_config as config
from model import get_model_and_tokenizer
from modeling.model_utils import repair_state_dict_keys

p = argparse.ArgumentParser()
p.add_argument('--load_path', required=True)
p.add_argument('--personas', nargs='+', default=['c06_food', 'c05_product', 'c02_architecture', 'c09_cinematic'])
p.add_argument('--seeds', type=int, nargs='+', default=list(range(8)))
p.add_argument('--guidance', type=float, default=4)
p.add_argument('--steps', type=int, default=50)
args = p.parse_args()
OUT = f'docs/results/S3_{os.path.basename(args.load_path.rstrip("/"))}'
os.makedirs(OUT, exist_ok=True)

config.load_path, config.do_compile, config.log_dir = args.load_path, False, OUT
model = get_model_and_tokenizer(config.transformer_model_path, config.device, config.dtype, config.seed, False, config)
model.eval()

# T0: is every checkpoint tensor in the model, exactly as saved?
sd = model.pipe.transformer.state_dict()
adapter = repair_state_dict_keys(torch.load(f'{args.load_path}/adapter.pt', map_location='cpu'))
# the saved LoRA file leaves out the adapter name ('default') that diffusers adds
lora = {'orig_transformer.' + k.replace('.weight', '.default.weight'): v
        for k, v in load_file(f'{args.load_path}/pytorch_lora_weights.safetensors').items()}
report, ok = [], True
for name, ckpt in [('adapter.pt', adapter), ('LoRA', lora)]:
    found = [k for k in ckpt if k in sd]
    equal = [k for k in found if sd[k].dtype == ckpt[k].dtype and torch.equal(sd[k].cpu(), ckpt[k])]
    report.append(f'T0 {name}: {len(ckpt)} tensors in file, {len(found)} in model, {len(equal)} identical')
    ok &= len(equal) == len(ckpt)
# and the other way: every trained tensor in the model must come from the checkpoint
trained = [k for k in sd if not k.startswith('orig_transformer.') or 'lora_' in k]
missing = [k for k in trained if k not in adapter and k not in lora]
report.append(f'T0 model: {len(trained)} trained tensors, {len(missing)} not in the checkpoint')
active = model.pipe.transformer.orig_transformer.active_adapters()
report.append(f'T0 active LoRA: {active}')
assert ok and not missing and active == ['default'], 'T0 failed:\n' + '\n'.join(report)

# Histories: 32 held-out images per persona, always the same ones. T2 mixes them, 8 from each persona.
from personas import load_personas, image_groups, select_personas
train = set(image_groups(['train']))
own = {n: random.Random(0).sample(sorted(p for p in m if p not in train), config.k)
       for n, m in load_personas(args.personas).items()}
mixed = [own[n][i] for i in range(config.k) for n in args.personas][:config.k]
histories = {}
for n in args.personas:
    histories['T1', n] = own[n], [5] * config.k
    histories['T2', n] = mixed, [5 if p in own[n] else 1 for p in mixed]
    histories['T3', n] = own[n], [1] * config.k

# Generate each history at each seed with Ryan's sampler, asking for a 5, then judge the images
from PIL import Image
from metrics.persona_judge import PersonaJudge
images = {}
for (test, n), (paths, scores) in histories.items():
    history = [Image.open(f'PAMELA/{p}').convert('RGB') for p in paths]
    for s in args.seeds:
        images[test, n, s] = model.do_qual_val(history, args.guidance, sample_scores=scores, target_scores=[5],
                                               seeds=[s], n_inference_steps=args.steps)[0]
del model
torch.cuda.empty_cache()
# judge among all the large personas, not just the tested ones: an image that looks like something else
# (a landscape, say) gets that label and counts as a miss, instead of being forced into one of the four
judge = PersonaJudge(sorted(set(select_personas()) | set(args.personas)), 'cuda')
report.append(f'Judge chooses among {len(judge.names)} personas')
keys = list(images)
verdict = dict(zip(keys, judge([images[k] for k in keys])[0]))

# Hit rate, and how often shuffling the personas within each seed does as well (exact, over all shuffles)
from itertools import permutations
P = args.personas
for test in ['T1', 'T2', 'T3']:
    hits = sum(verdict[test, n, s] == n for n in P for s in args.seeds)
    null = np.array([1.0])
    for s in args.seeds:
        counts = np.bincount([sum(verdict[test, n, s] == m for n, m in zip(P, perm)) for perm in permutations(P)],
                             minlength=len(P) + 1)
        null = np.convolve(null, counts / counts.sum())
    total = len(P) * len(args.seeds)
    what = 'retention of the disliked persona' if test == 'T3' else 'hit rate'
    report.append(f'{test} {what} {hits / total:.3f} ({hits}/{total}), exact permutation p {null[hits:].sum():.4f}')

# How much the image changes (mean pixel difference, 0-255) with the seed, the persona, and the score flip
from itertools import combinations
mad = lambda a, b: np.abs(np.asarray(a, float) - np.asarray(b, float)).mean()
for test in ['T1', 'T2', 'T3']:
    by_seed = [mad(images[test, n, a], images[test, n, b]) for n in P for a, b in combinations(args.seeds, 2)]
    by_persona = [mad(images[test, m, s], images[test, n, s]) for s in args.seeds for m, n in combinations(P, 2)]
    report.append(f'{test} image change: across seeds {np.mean(by_seed):.1f}, '
                  f'across personas at the same seed {np.mean(by_persona):.1f}')
flip = [mad(images['T1', n, s], images['T3', n, s]) for n in P for s in args.seeds]
report.append(f'T1 vs T3 (all 5 -> all 1), same persona and seed: {np.mean(flip):.1f}')

# Save the report, the verdicts, and one grid per test (rows = seeds, columns = personas)
from PIL import ImageDraw
open(f'{OUT}/validate.txt', 'w').write('\n'.join(report) + '\n')
with open(f'{OUT}/verdicts.tsv', 'w') as f:
    f.write('test\tpersona\tseed\tverdict\n')
    f.writelines(f'{t}\t{n}\t{s}\t{verdict[t, n, s]}\n' for t, n, s in keys)
T = 256
for test in ['T1', 'T2', 'T3']:
    grid = Image.new('RGB', (T * len(P), (T + 14) * len(args.seeds) + 14), 'white')
    d = ImageDraw.Draw(grid)
    for c, n in enumerate(P):
        d.text((c * T + 2, 1), n, fill='black')
        for r, s in enumerate(args.seeds):
            y = 14 + r * (T + 14)
            grid.paste(images[test, n, s].resize((T, T)), (c * T, y + 14))
            d.text((c * T + 2, y + 1), f'seed {s} -> {verdict[test, n, s]}', fill='black')
    grid.save(f'{OUT}/{test}_grid.jpg', quality=85)
print('\n'.join(report))
