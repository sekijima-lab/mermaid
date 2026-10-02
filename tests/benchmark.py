"""Cross-version CPU benchmark; Python 3.6 compatible, trusted bundled checkpoint only."""
import sys, os, json, hashlib
from pathlib import Path
import numpy as np
import torch
import hydra
import rdkit

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(root))
os.chdir(str(root))
hydra.utils.get_original_cwd = lambda: str(root)
from Utils.utils import VOCABULARY, parse_smiles, convert_smiles, read_smilesset
from Utils.reward import QEDReward, PenalizedLogPReward
from Model.model import RolloutNetwork
from Model.dataset import MolDataset
from Model.preprocess import parse_exhaustive_fragment, task_filter, task_modf

torch.set_num_threads(1)
torch.manual_seed(1729)
np.random.seed(1729)
checkpoint = root / 'ckpt/model-ep100.pth'
kwargs = {'map_location': 'cpu'}
if int(torch.__version__.split('.')[0]) >= 2:
    kwargs['weights_only'] = True
model = RolloutNetwork(len(VOCABULARY))
model.load_state_dict(torch.load(str(checkpoint), **kwargs))
model.eval()
fragments = read_smilesset('Data/preprocessed/fragments.smi')[:256]
tokens = [convert_smiles(parse_smiles('&' + s + '\n'), VOCABULARY, 's2i') for s in fragments]
dataset = MolDataset(tokens, 25)
batch = torch.stack([dataset[i][0] for i in range(len(dataset))])
lengths = [dataset[i][1] for i in range(len(dataset))]
with torch.no_grad():
    logits = model(batch, lengths)
    probabilities = torch.softmax(logits, dim=2)
np.savez_compressed(str(out / 'prediction.npz'), logits=logits.numpy(), probabilities=probabilities.numpy(), tokens=batch.numpy(), lengths=np.array(lengths))

# Evaluation mode suppresses stochastic dropout but still computes gradients.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
pred = model(batch[:32], lengths[:32])[:, :-1, :].contiguous().view(-1, len(VOCABULARY))
target = batch[:32, 1:].contiguous().view(-1)
loss = torch.nn.CrossEntropyLoss(ignore_index=0)(pred, target)
loss.backward()
gradients = {n: p.grad.detach().numpy().copy() for n,p in model.named_parameters()}
optimizer.step()
np.savez_compressed(str(out / 'training.npz'), **{'weight_' + n: p.detach().numpy() for n,p in model.named_parameters()}, **{'grad_' + n: a for n,a in gradients.items()})

smiles = read_smilesset('Data/input/sample_data.smi')[:512]
smiles += ['C1CCCCCCC1', 'C1CC2CCC1C2', 'invalid', 'CC(=O)Oc1ccccc1C(=O)O']
qed, plogp = QEDReward(), PenalizedLogPReward()
rewards = [[qed.reward(s), plogp.reward(s)] for s in smiles]
preprocessing = []
for s in smiles[:32]:
    fs = parse_exhaustive_fragment(s)
    preprocessing.append(sorted(set(f for x in fs for f in task_filter(x))))
summary = {'python':sys.version, 'torch':torch.__version__, 'rdkit':rdkit.__version__, 'numpy':np.__version__, 'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(), 'fragments':len(fragments), 'smiles':smiles, 'rewards':rewards, 'preprocessing':preprocessing, 'loss':float(loss.detach())}
(out/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps({k:v for k,v in summary.items() if k not in ['smiles','rewards','preprocessing']},indent=2))
