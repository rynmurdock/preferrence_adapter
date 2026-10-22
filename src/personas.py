'''
Simulated personas: naturally occurring image groups in PAMELA, found by clustering CLIP-L embeddings
A persona rates its own group's core images 5 and the other personas' 1.
'''
import json

PERSONAS_FILE = 'docs/results/S1_personas/personas.json'
SPLITS = ['train', 'val_seen', 'val_unseen', 'test_seen', 'test_unseen']


def load_personas(names=None, path=PERSONAS_FILE):
    '''{name: [core image paths]} for the chosen personas, or all of them if names is None.'''
    P = json.load(open(path))['personas']
    return {n: P[n]['members'] for n in (names or P)}


def image_groups(splits=SPLITS, root='PAMELA'):
    '''{image_path: PAMELA group label} over the given splits. The labels are noisy; we use them only to name groups.'''
    groups = {}
    for s in splits:
        for r in json.load(open(f'{root}/annotations/pamela_{s}.json')):
            groups[r['image_path']] = r['image_metadata']['group']
    return groups

def select_personas(min_heldout=32):
    '''Names of the personas with at least min_heldout held-out core images.'''
    train = set(image_groups(['train']))
    return [n for n, m in load_personas().items() if sum(p not in train for p in m) >= min_heldout]
