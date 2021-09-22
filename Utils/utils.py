import numpy as np
import math
from tqdm import tqdm
from copy import deepcopy
import warnings
warnings.filterwarnings('ignore')

from rdkit import DataStructs, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.six.moves import cPickle


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary
VOCABULARY = [
    'PAD',
    '#', '(', ')', '-', '/', '1', '2', '3', '4', '5', '6', '7', '8', '=',
    'Br', 'C', 'Cl', 'F', 'I', 'N', 'O', 'P', 'S',
    '[C@@H]', '[C@@]', '[C@H]', '[C@]', '[CH-]', '[CH2-]',
    '[N+]', '[N-]', '[NH+]', '[NH-]', '[NH2+]', '[NH3+]',
    '[O+]', '[O-]', '[OH+]',
    '[P+]', '[P@@H]', '[P@@]', '[P@]', '[PH+]', '[PH2]', '[PH]',
    '[S+]', '[S-]', '[S@@+]', '[S@@]', '[S@]', '[SH+]',
    '[n+]', '[n-]', '[nH+]', '[nH]', '[o+]', '[s+]', '\\', 'c', 'n', 'o', 's',
    '&', '\n'
]

atoms = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
         "[Sc]", "Ti", "V", "Cr", "[Mn]", "Fe", "[Co]", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb",
         "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "[Sn]", "Sb", "Te", "I", "Xe"]


class Node:
    def __init__(self):
        self.parent = None
        self.path = []
        self.depth = -100
        self.visit = 0
        self.children = []
        self.imm_score = 0
        self.cum_score = 0
        self.c = 1
        self.id = -1
        self.rollout_result = ("None", -1000)

    def add_Node(self, c):
        c.parent = self
        c.depth = self.depth + 1
        c.path = deepcopy(self.path)
        c.path.append(c.token)
        self.children.append(c)

    def calc_UCB(self):
        if self.visit == 0:
            ucb = 1e+6
        else:
            ucb = self.cum_score/self.visit + self.c*math.sqrt(2*math.log(self.parent.visit)/self.visit)
        return ucb

    def select_children(self):
        children_ucb = []
        for cn in self.children:
            children_ucb.append(cn.calc_UCB())
        max_ind = np.random.choice(np.where(np.array(children_ucb) == max(children_ucb))[0])
        return self.children[max_ind]


class RootNode(Node):
    def __init__(self, c=1/np.sqrt(2)):
        super().__init__()
        self.token = "&&"
        self.depth = 0
        self.path.append(self.token)
        self.c = c


class ParentNode(Node):
    def __init__(self, scacffold, c=1/np.sqrt(2)):
        super().__init__()
        self.original_smiles = scacffold
        self.token = "SCFD"
        self.c = c


class NormalNode(Node):
    def __init__(self, token, c=1/np.sqrt(2)):
        super().__init__()
        self.token = token
        self.c = c

    def remove_Node(self):
        self.parent.children.remove(self)


def convert_smiles(smiles, vocab, mode):
    """
    :param smiles:
    :param vocab: dict of tokens
    :param mode: s2i: string -> int
                 i2s: int -> string
    :return: converted smiles,
    """
    converted = []
    if mode == "s2i":
        for token in smiles:
            converted.append(vocab.index(token))
    elif mode == "i2s":
        for ind in smiles:
            converted.append(vocab[ind])
    return converted


def parse_smiles(smiles):
    parsed = []
    i = 0
    while i < len(smiles):
        asc = ord(smiles[i])
        if 64 < asc < 91:
            if i != len(smiles)-1 and smiles[i:i+2] in atoms:
                parsed.append(smiles[i:i+2])
                i += 2
            else:
                parsed.append(smiles[i])
                i += 1
        elif asc == 91:
            j = i
            while smiles[i] != "]":
                i += 1
            i += 1
            parsed.append(smiles[j:i])

        else:
            parsed.append(smiles[i])
            i += 1

    return parsed


def make_vocabulary(smiles_list, padding=True):
    vocab = []
    for smiles in smiles_list:
        vocab.extend(parse_smiles(smiles))
    vocab.append("&")
    vocab.append("\n")
    vocab = list(set(vocab))
    if padding:
        vocab.insert(0, "PAD")

    return vocab


def trans_infix_ringnumber(prefix, infix):
    stack = [0]
    for w in parse_smiles(prefix):
        try:
            if int(w) in stack:
                stack.remove(int(w))
            else:
                stack.append(int(w))
        except ValueError:
            pass
    max_num = max(stack)

    mod_infix = []
    for w in parse_smiles(infix):
        if w in [str(i) for i in range(10)]:
            mod_infix.append(str(int(w)+max_num))
        else:
            mod_infix.append(w)

    return "".join(mod_infix)


def read_vocabulary(path):
    with open(path) as f:
        vocabulary = []
        s = f.read()
        for w in s.split(","):
            if w is not "":
                vocabulary.append(w)

    return vocabulary


def read_smilesset(path):
    smiles_list = []
    with open(path) as f:
        for smiles in f:
            smiles_list.append(smiles.rstrip())

    return smiles_list


if __name__ == "__main__":
    smiles_list = read_smilesset("Data/250k_rndm_zinc_drugs_clean.smi")
    vocab = []
    for smiles in tqdm(smiles_list):
        p = parse_smiles(smiles)
        vocab.extend(p)

    vocab = list(set(vocab))
    vocab.sort()
    print(vocab)


