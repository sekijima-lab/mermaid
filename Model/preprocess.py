import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.utils import *


def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]


def task_filter(smiles):
    part = []
    # try:
    #     sc, fr = smiles.split(".")
    # except ValueError:
    #     return part

    mol_sc = Chem.MolFromSmiles(smiles)
    mol_fr = Chem.MolFromSmiles("C" + smiles + "C")
    # if mol_sc is None or mol_fr is None:
    if mol_sc is not None and mol_fr is not None:
        # part.append("(" + smiles + ")")
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
    with mp.Pool(mp.cpu_count()-1) as p:
        part = p.map(task_filter, smiles_list)
    part = list(itertools.chain.from_iterable(part))
    part = list(set(part))

    with mp.Pool(mp.cpu_count()-1) as p:
        part = p.map(task_modf, part)
    part = list(itertools.chain.from_iterable(part))
    part = list(set(part))

    return part


def parse_exhaustive_fragment(smiles, maximum_length=20):
    fragments = []
    p = parse_smiles(smiles)
    for l in range(maximum_length):
        for i in range(len(p)+1):
            fr = "".join(p[i:i + l])
            fragments.append(fr)

    return fragments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="Data/250k_rndm_zinc_drugs_clean.smi")
    parser.add_argument("--out_dir", type=str, default="Data/")
    parser.add_argument("--traintest_ratio", type=int, default=0.8)

    args = parser.parse_args()

    smiles_list = read_smilesset(args.datapath)
    random.shuffle(smiles_list)
    sp = int(len(smiles_list)*args.traintest_ratio)
    traintest_list = smiles_list[:sp]
    validation_list = smiles_list[sp:]

    fragment_list = []
    for smiles in tqdm(traintest_list):
        fragments = parse_exhaustive_fragment(smiles)
        fragment_list.extend(fragments)
    fragment_list = list(set(fragment_list))

    fragment_list = valid_parse(fragment_list)

    with open(f"{args.out_dir}/fragments.smi", mode="w") as f:
        for s in fragment_list:
            f.write(s+"\n")

    with open(f"{args.out_dir}/validation.smi", mode="w") as f:
        for s in validation_list:
            f.write(s+"\n")




