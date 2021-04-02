import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Utils.sascore import calculateScore
from Utils.utils import *


class Reward:
    def __init__(self):
        return

    def reward(self, smiles):
        raise NotImplementedError()


class PenalizedLogPReward(Reward):
    def __init__(self, *args, **kwargs):
        super(PenalizedLogPReward, self).__init__(*args, **kwargs)
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

        else:
            score = -100

        return score


class QEDReward(Reward):
    def __init__(self, *args, **kwargs):
        super(QEDReward, self).__init__(*args, **kwargs)

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


class SAReward(Reward):
    def __init__(self):
        super(SAReward, self).__init__()

    def reward(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return calculateScore(mol)
        else:
            return -100


class ConstReward:
    # Tanimoto Similarity based on ECFP4
    def __init__(self, reward_module, threshold=0.6):
        self.reward_module = reward_module
        self.th = threshold
        self.source_fp = None

    def setInitial(self, init_smiles, threshold):
        self.source_fp = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(init_smiles), 2)
        self.th = threshold

    def reward(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        try:
            target_fp = AllChem.GetMorganFingerprint(mol, 2)
            sim = DataStructs.TanimotoSimilarity(self.source_fp, target_fp)
        except ValueError:
            sim = 0

        score = -100
        if sim > self.th:
            try:
                if mol is not None:
                    score = self.reward_module.reward(smiles)
            except:
                pass

        return score


class TargetingReward:
    def __init__(self, reward_module, ranges=(0, 1)):
        self.reward_module = reward_module
        self.ranges = ranges

    def reward(self, smiles):
        score = self.reward_module.reward(smiles)

        if self.ranges[0] <= score <= self.ranges[1]:
            return 1
        else:
            return -min(abs(score-self.ranges[0]), abs(score-self.ranges[1]))
