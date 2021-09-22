import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from torch.utils.data import DataLoader, Dataset

from Utils.utils import read_smilesset, parse_smiles, convert_smiles
from Utils.utils import VOCABULARY


class MolDataset(Dataset):
    def __init__(self, smiles_list, seq_len):
        super(MolDataset, self).__init__()
        self.smiles_list = smiles_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, item):
        x = self.smiles_list[item]
        x_len = len(x)
        x = torch.tensor(x, dtype=torch.long)
        buf = torch.zeros(self.seq_len, dtype=torch.long)
        buf[:len(x)] = x
        x = buf
        # x = torch.nn.functional.one_hot(x.to(torch.int64), num_classes=len(self.vocab))

        return x, x_len


def main():
    smiles_list = read_smilesset("Data/sample_data.smi")
    vocab = VOCABULARY

    smiles_list = [parse_smiles(smiles) for smiles in smiles_list]
    smiles_list = [convert_smiles(s, vocab, mode="s2i") for s in smiles_list]

    dataset = MolDataset(smiles_list, 80)
    mol_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)

    for i, (x, x_len) in enumerate(mol_loader):
        print(x.shape)
        break


if __name__ == "__main__":
    main()

