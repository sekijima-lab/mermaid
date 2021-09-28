import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import hydra
from config.config import cs
from omegaconf import DictConfig
import torch
import time
import warnings
warnings.filterwarnings('ignore')

import rdkit.Chem as Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')
from rdkit.six.moves import cPickle

import torch.nn.functional as F

from Model.model import RolloutNetwork
from Utils.utils import read_smilesset, parse_smiles, convert_smiles, RootNode, ParentNode, NormalNode, \
    trans_infix_ringnumber
from Utils.utils import VOCABULARY
from Utils.reward import getReward

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MCTS(object):
    def __init__(self, init_smiles, model, vocab, Reward, max_seq=81, c=1, num_prll=256, limit=5, step=0, n_valid=0,
                 n_invalid=0, sampling_max=False, max_r=-1000):
        self.init_smiles = parse_smiles(init_smiles.rstrip("\n"))
        self.model = model
        self.vocab = vocab
        self.Reward = Reward
        self.max_seq = max_seq
        self.valid_smiles = {}
        self.c = c
        self.count = 0
        self.ub_prll = num_prll
        self.limit = np.sum([len(self.init_smiles)+1-i for i in range(limit)])
        self.sq = set([s for s in self.vocab if "[" in s])
        self.max_score = max_r
        self.step = step
        self.n_valid = n_valid
        self.n_invalid = n_invalid
        self.sampling_max = sampling_max
        self.total_nodes = 0

    def select(self):
        raise NotImplementedError()

    def expand(self):
        raise NotImplementedError()

    def simulate(self):
        raise NotImplementedError()

    def backprop(self):
        raise NotImplementedError()

    def search(self, n_step):
        raise NotImplementedError()


class ParseSelectMCTS(MCTS):
    def __init__(self, *args, **kwargs):
        super(ParseSelectMCTS, self).__init__(*args, **kwargs)
        self.root = RootNode()
        self.current_node = None
        self.next_token = {}
        self.rollout_result = {}
        self.l_replace = int(len(self.init_smiles)/4)

    def select(self):
        """
        search for the node with no child nodes and maximum UCB score
        """
        self.current_node = self.root
        while len(self.current_node.children) != 0:
            self.current_node = self.current_node.select_children()
            if self.current_node.depth+1 > self.max_seq:
                tmp = self.current_node
                # update
                while self.current_node is not None:
                    self.current_node.cum_score += -1
                    self.current_node.visit += 1
                    self.current_node = self.current_node.parent
                tmp.remove_Node()

                self.current_node = self.root

    def expand(self, epsilon=0.1, loop=10, gamma=0.90):
        """

        """
        # Preparation of prediction using RNN model, list -> tensor
        x = np.zeros([1, self.max_seq])
        c_path = convert_smiles(self.current_node.path[2:], self.vocab, mode="s2i")
        x[0, :len(c_path)] = c_path
        x = torch.tensor(x, dtype=torch.long)
        x_len = [len(c_path)]

        # Predict the probabilities of next token following current node
        with torch.no_grad():
            y = self.model(x, x_len)
            y = F.softmax(y, dim=2)
            y = y.to('cpu').detach().numpy().copy()
            y = np.array(y[0, len(self.current_node.path)-3, :])
            y = np.log(y)
            prob = np.exp(y) / np.sum(np.exp(y))

        # Sampling next token based on the probabilities
        self.next_token = {}
        while len(self.next_token) == 0:
            for j in range(loop):
                if np.random.rand() > epsilon * (gamma ** len(self.current_node.path)):
                    ind = np.random.choice(range(len(prob)), p=prob)
                else:
                    ind = np.random.randint(len(self.vocab))
                self.next_token[self.vocab[ind]] = 0
            if self.current_node.depth == 1:
                self.next_token["("] = 0
        self.check()

        print("".join(self.current_node.path[2:]), len(self.next_token))
        print(self.next_token.keys())

    def check(self):
        if "\n" in self.next_token.keys():
            tmp_node = self.current_node
            while tmp_node.depth != 1:
                tmp_node = tmp_node.parent
            original_smiles = tmp_node.original_smiles
            pref, suf = original_smiles.split("*")
            inf = "".join(self.current_node.path[3:])
            smiles_concat = pref + trans_infix_ringnumber(pref, inf) + suf

            score = self.Reward.reward(smiles_concat)

            self.max_score = max(self.max_score, score)
            self.next_token.pop("\n")
            if score > -100:
                self.valid_smiles["%d:%s" % (-self.step, smiles_concat)] = score
                print(score, smiles_concat)
                self.max_score = max(self.max_score, score)
                self.n_valid += 1
            else:
                self.n_invalid += 1

        if len(self.next_token) < 1:
            self.current_node.cum_score = -100000
            self.current_node.visit = 100000
            self.current_node.remove_Node()

    def simulate(self):
        tmp_node = self.current_node
        while tmp_node.depth != 1:
            tmp_node = tmp_node.parent
        original_smiles = tmp_node.original_smiles
        pref, suf = original_smiles.split("*")
        self.rollout_result = {}

        #######################################

        l = len(self.current_node.path)
        part_smiles = [[] for i in range(len(self.next_token))]
        x = np.zeros([len(self.next_token), self.max_seq])
        x_len = []
        for i, k in enumerate(self.next_token.keys()):
            part_smiles[i].extend(self.current_node.path[2:])
            part_smiles[i].append(k)
            x[i, :len(part_smiles[i])] = convert_smiles(part_smiles[i], self.vocab, mode="s2i")
            x_len.append(len(part_smiles[i]))
        x = torch.tensor(x, dtype=torch.long)

        is_terminator = [True]*len(self.next_token)
        step = 0

        while np.sum(is_terminator) > 0 and step+l < self.max_seq-1:
            with torch.no_grad():
                y = self.model(x, x_len)
                y = F.softmax(y, dim=2)
                y = y.to('cpu').detach().numpy().copy()
                prob = y[:, step+l-2, :]

            if self.sampling_max:
                ind = np.argmax(prob, axis=1)
            else:
                ind = [np.random.choice(range(len(self.vocab)), p=prob[i]) for i in range(len(self.next_token))]

            for i in range(len(x_len)):
                x_len[i] += 1

            for i in range(len(self.next_token)):
                x[i, step+l-1] = ind[i]
                if is_terminator[i] and ind[i] == self.vocab.index("\n"):
                    is_terminator[i] = False
                    inf = "".join(convert_smiles(x[i, 1:step+l-1], self.vocab, mode="i2s"))
                    smiles_concat = pref + trans_infix_ringnumber(pref, inf) + suf

                    score = self.Reward.reward(smiles_concat)

                    self.next_token[list(self.next_token.keys())[i]] = score
                    self.rollout_result[list(self.next_token.keys())[i]] = (smiles_concat, score)
                    if score > self.Reward.vmin:
                        # self.valid_smiles[smiles_concat] = score
                        self.valid_smiles["%d:%s" % (self.step, smiles_concat)] = score
                        self.max_score = max(self.max_score, score)
                        print(score, smiles_concat)
                        self.n_valid += 1
                    else:
                        # print("NO", smiles_concat)
                        self.n_invalid += 1
            step += 1

    def backprop(self):
        for i, key in enumerate(self.next_token.keys()):
            child = NormalNode(key, c=self.c)
            child.id = self.total_nodes
            self.total_nodes += 1
            try:
                child.rollout_result = self.rollout_result[key]
            except KeyError:
                child.rollout_result = ("Termination", -10000)
            self.current_node.add_Node(child)
        max_reward = max(self.next_token.values())
        # self.max_score = max(self.max_score, max_reward)
        while self.current_node is not None:
            self.current_node.visit += 1
            self.current_node.cum_score += max_reward/(1+abs(max_reward))
            self.current_node.imm_score = max(self.current_node.imm_score, max_reward/(1+abs(max_reward)))
            self.current_node = self.current_node.parent

    def search(self, n_step, epsilon=0.1, loop=10, gamma=0.90, rep_file=None):
        self.set_repnode(rep_file=rep_file)

        while self.step < n_step:
            self.step += 1
            print("--- step %d ---" % self.step)
            print("MAX_SCORE:", self.max_score)
            if self.n_valid+self.n_invalid == 0:
                valid_rate = 0
            else:
                valid_rate = self.n_valid/(self.n_valid+self.n_invalid)
            print("Validity rate:", valid_rate)
            self.select()
            self.expand(epsilon=epsilon, loop=loop, gamma=gamma)
            if len(self.next_token) != 0:
                self.simulate()
                self.backprop()

    def set_repnode(self, rep_file=None):
        if len(rep_file) > 0:
            for smiles in read_smilesset(hydra.utils.get_original_cwd()+rep_file):
                n = ParentNode(smiles)
                self.root.add_Node(n)
                c = NormalNode("&")
                n.add_Node(c)
        else:
            for i in range(self.l_replace+1):
                for j in range(len(self.init_smiles)-i+1):
                    infix = self.init_smiles[j:j+i]
                    prefix = "".join(self.init_smiles[:j])
                    suffix = "".join(self.init_smiles[j + i:])

                    sc = prefix + "(*)" + suffix
                    mol_sc = Chem.MolFromSmiles(sc)
                    if mol_sc is not None:
                        n = ParentNode(prefix + "(*)" + suffix)
                        self.root.add_Node(n)
                        c = NormalNode("&")
                        n.add_Node(c)

    def save_tree(self, dir_path):
        for i in range(len(self.root.children)):
            stack = []
            stack.extend(self.root.children[i].children)
            sc = self.root.children[i].original_smiles
            score = [self.root.children[i].cum_score]
            ids = [-1]
            parent_id = [-1]
            children_id = [[c.id for c in self.root.children[i].children]]
            infix = [sc]
            rollout_smiles = ["Scaffold"]
            rollout_score = [-10000]

            while len(stack) > 0:
                c = stack.pop(-1)
                for gc in c.children:
                    stack.append(gc)

                # save information
                score.append(c.cum_score)
                ids.append(c.id)
                parent_id.append(c.parent.id)
                ch_id = [str(gc.id) for gc in c.children]
                children_id.append(",".join(ch_id))
                infix.append("".join(c.path))
                rollout_smiles.append(c.rollout_result[0])
                rollout_score.append(c.rollout_result[1])

            df = pd.DataFrame(columns=["ID", "Score", "P_ID", "C_ID", "Infix", "Rollout_SMILES", "Rollout_Score"])
            df["ID"] = ids
            df["Score"] = score
            df["P_ID"] = parent_id
            df["C_ID"] = children_id
            df["Infix"] = infix
            df["Rollout_SMILES"] = rollout_smiles
            df["Rollout_Score"] = rollout_score

            df.to_csv(dir_path+f"/tree{i}.csv", index=False)


@hydra.main(config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    """--- constant ---"""
    vocab = VOCABULARY

    """--- input smiles ---"""
    start_smiles_list = read_smilesset(hydra.utils.get_original_cwd() + cfg["mcts"]["in_smiles_file"])

    for n, start_smiles in enumerate(start_smiles_list):
        n_valid = 0
        n_invalid = 0
        gen = {}
        mcts = None

        """--- MCTS ---"""
        model = RolloutNetwork(len(vocab))
        model_ver = cfg["mcts"]["model_ver"]
        model.load_state_dict(torch.load(hydra.utils.get_original_cwd() + cfg["mcts"]["model_dir"]
                                         + f"model-ep{model_ver}.pth",  map_location=torch.device('cpu')))

        reward = getReward(name=cfg["mcts"]["reward_name"], init_smiles=start_smiles)

        input_smiles = start_smiles
        start = time.time()
        for i in range(cfg["mcts"]["n_iter"]):
            mcts = ParseSelectMCTS(input_smiles, model=model, vocab=vocab, Reward=reward,
                                   max_seq=cfg["mcts"]["seq_len"], step=cfg["mcts"]["n_step"] * i,
                                   n_valid=n_valid, n_invalid=n_invalid, c=cfg["mcts"]["ucb_c"], max_r=reward.max_r)
            mcts.search(n_step=cfg["mcts"]["n_step"] * (i + 1), epsilon=0, loop=10, rep_file=cfg["mcts"]["rep_file"])
            reward.max_r = mcts.max_score
            n_valid += mcts.n_valid
            n_invalid += mcts.n_invalid
            gen = sorted(mcts.valid_smiles.items(), key=lambda x: x[1], reverse=True)
            input_smiles = gen[0][0].split(":")[1]
        end = time.time()
        print("Elapsed Time: %f" % (end-start))

        generated_smiles = pd.DataFrame(columns=["SMILES", "Reward", "Imp", "MW", "step"])
        start_reward = reward.reward(start_smiles)
        for kv in mcts.valid_smiles.items():
            step, smi = kv[0].split(":")
            step = int(step)

            try:
                w = Descriptors.MolWt(Chem.MolFromSmiles(smi))
            except:
                w = 0

            generated_smiles.at[smi.rstrip('\n'), "SMILES"] = smi
            generated_smiles.at[smi.rstrip('\n'), "Reward"] = kv[1]
            generated_smiles.at[smi.rstrip('\n'), "Imp"] = kv[1] - start_reward
            generated_smiles.at[smi.rstrip('\n'), "MW"] = w
            generated_smiles.at[smi.rstrip('\n'), "step"] = step

        generated_smiles = generated_smiles.sort_values("Reward", ascending=False)
        generated_smiles.to_csv(hydra.utils.get_original_cwd() +
                                cfg["mcts"]["out_dir"] + "No-{:04d}-{}.csv".format(n, start_smiles), index=False)


if __name__ == "__main__":
    main()


