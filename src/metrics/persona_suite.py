'''Persona validation suite, shared by scripts/validate.py (full) and src/train.py (light, at each validation).'''
import random
from itertools import permutations
import numpy as np
import torch
from PIL import Image
from personas import load_personas, image_groups, select_personas
from metrics.persona_judge import PersonaJudge


def histories(personas, k):
    '''{(test, persona): (image paths, scores)}, from held-out images only.
    T1: the persona's own k images, all rated 5. T2: one mixed history, 5 for the persona's own, 1 for the rest.
    T3: T1's images, all rated 1.'''
    train = set(image_groups(['train']))
    own = {n: random.Random(0).sample(sorted(p for p in m if p not in train), k)
           for n, m in load_personas(personas).items()}
    mixed = [own[n][i] for i in range(k) for n in personas][:k]
    H = {}
    for n in personas:
        H['T1', n] = own[n], [5] * k
        H['T2', n] = mixed, [5 if p in own[n] else 1 for p in mixed]
        H['T3', n] = own[n], [1] * k
    return H


def generate(model, H, seeds, guidance=4, steps=50):
    '''{(test, persona, seed): image}, with the repo's sampler, asking for a 5'''
    images = {}
    with torch.no_grad():
        for (test, n), (paths, scores) in H.items():
            history = [Image.open(f'PAMELA/{p}').convert('RGB') for p in paths]
            for s in seeds:
                images[test, n, s] = model.do_qual_val(history, guidance, sample_scores=scores, target_scores=[5],
                                                       seeds=[s], n_inference_steps=steps)[0]
    return images


def make_judge(personas, device):
    '''Judge among all the large personas, so an image that looks like something else counts as a miss'''
    return PersonaJudge(sorted(set(select_personas()) | set(personas)), device)


def hits_and_p(verdict, personas, seeds, test):
    '''(hits, total, exact p): how often shuffling the personas within each seed does at least as well'''
    hits = sum(verdict[test, n, s] == n for n in personas for s in seeds)
    null = np.array([1.0])
    for s in seeds:
        counts = np.bincount([sum(verdict[test, n, s] == m for n, m in zip(personas, perm))
                              for perm in permutations(personas)], minlength=len(personas) + 1)
        null = np.convolve(null, counts / counts.sum())
    return hits, len(personas) * len(seeds), null[hits:].sum()
