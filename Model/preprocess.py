import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import itertools
import collections
import random
from tqdm import tqdm
import multiprocessing as mp
import warnings
import hydra
from config.config import cs
from omegaconf import DictConfig
warnings.filterwarnings('ignore')

import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.six.moves import cPickle

from Utils.utils import parse_smiles, read_smilesset


def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]


def task_filter(smiles):
    part = []

    mol_fr = Chem.MolFromSmiles("C(" + smiles + ")C")
    if mol_fr is not None:
        part.append(smiles)

    return part


def task_modf(smiles):
    c = collections.Counter(smiles)
    s = smiles
    n = 0
    for i in range(1, 10):
        if c[str(i)] > 0:
            n = i
            break

    if n > 0:
        ring = [str(i) for i in range(n, 10)]
        s = parse_smiles(s)
        for i, w in enumerate(s):
            if w in ring:
                s[i] = str(int(w)-n+1)
        s = "".join(s)
    return [s]


def valid_parse(smiles_list):
    # Filtering valid partial SMILES
    with mp.Pool(mp.cpu_count()) as p:
        part = p.map(task_filter, smiles_list)
    part = list(itertools.chain.from_iterable(part))
    part = list(set(part))

    # Adjust the ring number
    with mp.Pool(mp.cpu_count()-1) as p:
        part = p.map(task_modf, part)
    part = list(itertools.chain.from_iterable(part))
    part = list(set(part))

    return part


def parse_exhaustive_fragment(smiles, max_length=20):
    fragments = []
    p = parse_smiles(smiles)
    for l in range(max_length):
        for i in range(len(p)+1):
            fr = "".join(p[i:i + l])
            fragments.append(fr)

    return fragments


@hydra.main(config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    # Loading data
    smiles_list = read_smilesset(hydra.utils.get_original_cwd()+cfg["prep"]["datapath"])
    random.shuffle(smiles_list)
    sp = int(len(smiles_list) * cfg["prep"]["ratio"])
    train_list = smiles_list[:sp]
    validation_list = smiles_list[sp:]

    # Preprocessing for learning
    fragment_list = []
    for smiles in tqdm(train_list):
        fragments = parse_exhaustive_fragment(smiles, max_length=cfg["prep"]["max_len"])
        fragment_list.extend(fragments)
    fragment_list = list(set(fragment_list))

    fragment_list = valid_parse(fragment_list)

    # Saving data
    outpath = hydra.utils.get_original_cwd()+cfg["prep"]["outdir"]
    with open(f"{outpath}/fragments.smi", mode="w") as f:
        for s in fragment_list:
            f.write(s + "\n")

    with open(f"{outpath}/validation.smi", mode="w") as f:
        for s in validation_list:
            f.write(s + "\n")


if __name__ == "__main__":
    main()




