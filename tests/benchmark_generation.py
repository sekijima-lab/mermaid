import sys,os,json
from pathlib import Path
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root));os.chdir(str(root))
import numpy as np,torch,hydra
from Utils.utils import VOCABULARY,read_smilesset
from Utils.reward import PenalizedLogPReward
from Model.model import RolloutNetwork
from Generator.mcts import ParseSelectMCTS
hydra.utils.get_original_cwd=lambda:str(root)
torch.set_num_threads(1)
kw={'map_location':'cpu'}
if int(torch.__version__.split('.')[0])>=2:kw['weights_only']=True
model=RolloutNetwork(len(VOCABULARY));model.load_state_dict(torch.load(str(root/'ckpt/model-ep100.pth'),**kw));model.eval()
result=[]
for smi in ['CCO',read_smilesset('Data/input/init_smiles.smi')[0]]:
 for seed in [0,1,2]:
    np.random.seed(seed);torch.manual_seed(seed)
    m=ParseSelectMCTS(smi,model=model,vocab=VOCABULARY,Reward=PenalizedLogPReward(),max_seq=25)
    m.search(n_step=3,epsilon=0,loop=3,rep_file='')
    result.append({'input':smi,'seed':seed,'generated':m.valid_smiles,'n_valid':m.n_valid,'n_invalid':m.n_invalid})
(out/'generation.json').write_text(json.dumps(result,indent=2));print('DONE',len(result))
