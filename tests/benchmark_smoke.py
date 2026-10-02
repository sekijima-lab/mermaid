"""Exercise actual training and generation functions against isolated output paths."""
import os, sys, json, tempfile, contextlib
from pathlib import Path
root=Path(sys.argv[1]).resolve(); out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(root));os.chdir(str(root))
import numpy as np, torch, hydra, mlflow
torch.set_num_threads(1);torch.manual_seed(1729);np.random.seed(1729)
from Model.train import train
from config.config import Config
from omegaconf import OmegaConf
from Utils.utils import VOCABULARY
from Model.model import RolloutNetwork
from Generator.mcts import ParseSelectMCTS
from Utils.reward import QEDReward
scratch=out/'scratch';scratch.mkdir(exist_ok=True)
for name in ['ckpt','logs']:(scratch/name).mkdir(exist_ok=True)
frags=(root/'Data/preprocessed/fragments.smi').read_text().splitlines()[:40]
(scratch/'fragments.smi').write_text('\n'.join(frags)+'\n')
hydra.utils.get_original_cwd=lambda:str(scratch)
os.environ['MLFLOW_TRACKING_URI']=(scratch/'mlruns').as_uri()
os.environ['MLFLOW_ALLOW_FILE_STORE']='true'
mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
cfg=OmegaConf.structured(Config)
cfg.train.datapath='/fragments.smi';cfg.train.epoch=1;cfg.train.save_step=1;cfg.train.batch_size=8
train(cfg)
client=mlflow.tracking.MlflowClient()
runs=client.search_runs(['0']);assert len(runs)==1
run=runs[0];assert run.info.status=='FINISHED'
assert run.data.params=={'batch_size':'8','lr':'0.0001','epoch':'1'}
assert set(run.data.metrics)=={'Train total loss','Test total loss'}
assert all(np.isfinite(v) for v in run.data.metrics.values())
assert client.get_metric_history(run.info.run_id,'Train total loss')[0].step==1
artifacts=[a.path for a in client.list_artifacts(run.info.run_id)]
assert artifacts==['log.txt']
download=client.download_artifacts(run.info.run_id,'log.txt',str(out));assert Path(download).read_text()=='log.txt'
kw={'map_location':'cpu'}
if int(torch.__version__.split('.')[0])>=2:kw['weights_only']=True
model=RolloutNetwork(len(VOCABULARY));model.load_state_dict(torch.load(str(root/'ckpt/model-ep100.pth'),**kw))
# Preserve original generation mode (dropout active); numerical equality is tested separately.
hydra.utils.get_original_cwd=lambda:str(root)
search=ParseSelectMCTS('CCO',model=model,vocab=VOCABULARY,Reward=QEDReward(),max_seq=25,num_prll=8)
search.search(n_step=3,epsilon=0,loop=3,rep_file='')
assert search.step==3 and search.n_valid>0
search.save_tree(str(scratch))
summary={'torch':torch.__version__,'mlflow':mlflow.__version__,'params':run.data.params,'metrics':run.data.metrics,'artifact_contents':Path(download).read_text(),'generated':search.valid_smiles,'n_valid':search.n_valid,'n_invalid':search.n_invalid}
(out/'smoke.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
