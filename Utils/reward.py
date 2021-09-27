import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import warnings
warnings.filterwarnings('ignore')

import rdkit.Chem as Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.six.moves import cPickle
from rdkit.Chem import AllChem, QED, DataStructs, Descriptors

from Utils.sascore import calculateScore


def getReward(name, init_smiles):
    if name == "QED":
        return QEDReward()
    elif name == "PLogP":
        return PenalizedLogPReward()


class Reward:
    def __init__(self):
        self.vmin = -100
        self.max_r = -10000
        return

    def reward(self, smiles):
        raise NotImplementedError()


class PenalizedLogPReward(Reward):
    def __init__(self, *args, **kwargs):
        super(PenalizedLogPReward, self).__init__(*args, **kwargs)
        self.vmin = -100
        return

    def reward(self, smiles):
        """
            This code is obtained from https://drive.google.com/drive/folders/1FmYWcT8jDrwZlzPbmMpRhulb9OKTDWJL
            , which is a part of GraphAF program done by Chence Shi.
            Reward that consists of log p penalized by SA and # long cycles,
            as described in (Kusner et al. 2017). Scores are normalized based on the
            statistics of 250k_rndm_zinc_drugs_clean.smi dataset
            :param mol: rdkit mol object
            :return: float
            """
        # normalization constants, statistics from 250k_rndm_zinc_drugs_clean.smi
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        SA_mean = -3.0525811293166134
        SA_std = 0.8335207024513095
        cycle_mean = -0.0485696876403053
        cycle_std = 0.2860212110245455

        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                log_p = Descriptors.MolLogP(mol)
                SA = -calculateScore(mol)

                # cycle score
                cycle_list = nx.cycle_basis(nx.Graph(
                    Chem.rdmolops.GetAdjacencyMatrix(mol)))
                if len(cycle_list) == 0:
                    cycle_length = 0
                else:
                    cycle_length = max([len(j) for j in cycle_list])
                if cycle_length <= 6:
                    cycle_length = 0
                else:
                    cycle_length = cycle_length - 6
                cycle_score = -cycle_length

                normalized_log_p = (log_p - logP_mean) / logP_std
                normalized_SA = (SA - SA_mean) / SA_std
                normalized_cycle = (cycle_score - cycle_mean) / cycle_std
                score = normalized_log_p + normalized_SA + normalized_cycle
            except ValueError:
                score = self.vmin
        else:
            score = self.vmin

        return score


class QEDReward(Reward):
    def __init__(self, *args, **kwargs):
        super(QEDReward, self).__init__(*args, **kwargs)
        self.vmin = 0

    def reward(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        try:
            if mol is not None:
                score = QED.qed(mol)
            else:
                score = -1
        except:
            score = -1

        return score

