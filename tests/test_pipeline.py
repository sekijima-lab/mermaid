import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Model.model import RolloutNetwork
from Utils.utils import VOCABULARY
from config.config import Config

torch.set_num_threads(1)


class PipelineTests(unittest.TestCase):
    def test_independent_configuration(self):
        a, b = Config(), Config()
        a.train.lr = 1
        self.assertEqual(b.train.lr, 0.0001)

    def test_checkpoint_prediction_matches_legacy(self):
        model = RolloutNetwork(len(VOCABULARY))
        model.load_weights(ROOT / 'ckpt/model-ep100.pth')
        model.eval()
        with np.load(ROOT / 'validation/baseline/prediction.npz', allow_pickle=False) as ref:
            with torch.no_grad():
                y = model(torch.tensor(ref['tokens']), ref['lengths'].tolist()).numpy()
            np.testing.assert_allclose(y, ref['logits'], rtol=0, atol=0.0001)
            np.testing.assert_array_equal(y.argmax(2), ref['logits'].argmax(2))

    def test_checkpoint_rejects_general_pickle_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'non_tensor.pth'
            torch.save({'non_tensor': object()}, path)
            with self.assertRaises(pickle.UnpicklingError):
                RolloutNetwork(len(VOCABULARY)).load_weights(path)

    def test_preprocessing_training_cli_and_tracking(self):
        import mlflow
        os.environ['MLFLOW_ALLOW_FILE_STORE'] = 'true'
        with tempfile.TemporaryDirectory() as tmp:
            scratch = Path(tmp)
            (scratch / 'input.smi').write_text('CCO\nCCN\nCCCO\nCCCN\nCCOC\nCCNC\n')
            for name in ['preprocessed', 'ckpt', 'logs']:
                (scratch / name).mkdir()
            prefix = '/' + os.path.relpath(scratch, ROOT)
            env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MLFLOW_DISABLE_AGENT_HINT='1')
            env.pop('MLFLOW_TRACKING_URI', None)
            commands = [
                ['Model/preprocess.py', 'prep.datapath='+prefix+'/input.smi', 'prep.outdir='+prefix+'/preprocessed', 'prep.max_len=4', 'hydra.run.dir='+str(scratch/'prep-run')],
                ['Model/train.py', 'train.datapath='+prefix+'/preprocessed/fragments.smi', 'train.epoch=1', 'train.batch_size=4', 'train.save_step=1', 'train.ckptdir='+prefix+'/ckpt/', 'train.log_dir='+prefix+'/logs/', 'hydra.run.dir='+str(scratch/'train-run')],
            ]
            for command in commands:
                result = subprocess.run([sys.executable] + command, cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue((scratch/'preprocessed/validation.smi').exists())
            model = RolloutNetwork(len(VOCABULARY))
            model.load_weights(scratch/'ckpt/model-ep1.pth')
            client = mlflow.tracking.MlflowClient(tracking_uri=(scratch/'train-run/mlruns').as_uri())
            runs = client.search_runs(['0'])
            self.assertEqual(len(runs), 1)
            run = runs[0]
            self.assertEqual(run.info.status, 'FINISHED')
            self.assertEqual(run.data.params, {'batch_size':'4', 'epoch':'1', 'lr':'0.0001'})
            self.assertEqual(set(run.data.metrics), {'Train total loss','Test total loss'})
            self.assertTrue(all(np.isfinite(v) for v in run.data.metrics.values()))
            self.assertEqual(client.get_metric_history(run.info.run_id,'Train total loss')[0].step, 1)
            path = client.download_artifacts(run.info.run_id, 'log.txt', str(scratch))
            self.assertEqual(Path(path).read_text(), 'log.txt')

    def test_controlled_generation_matches_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, PYTHONHASHSEED='0', MLFLOW_DISABLE_AGENT_HINT='1')
            result = subprocess.run([sys.executable, str(ROOT/'tests/benchmark_generation.py'), str(ROOT), tmp], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertEqual(json.loads((Path(tmp)/'generation.json').read_text()), json.loads((ROOT/'validation/baseline/generation.json').read_text()))


if __name__ == '__main__':
    unittest.main()
