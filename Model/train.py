import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import hydra
from config.config import cs
from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow import log_metric, log_param, log_artifacts
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Model.dataset import MolDataset
from Model.model import RolloutNetwork
from Utils.utils import read_smilesset, parse_smiles, convert_smiles
from Utils.utils import VOCABULARY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(cfg):
    vocab = VOCABULARY
    fragment_list = read_smilesset(hydra.utils.get_original_cwd()+cfg["train"]["datapath"])
    fragment_list = [convert_smiles(parse_smiles("&" + s.rstrip("\n") + "\n"), vocab, mode="s2i") for s in fragment_list]

    sp = int(len(fragment_list) * (1 - cfg["train"]["ratio"]))
    train_dataset = MolDataset(fragment_list[:sp], cfg["train"]["seq_len"])
    valid_dataset = MolDataset(fragment_list[sp:], cfg["train"]["seq_len"])
    train_loader = DataLoader(train_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg["train"]["batch_size"], shuffle=True)

    model = RolloutNetwork(vocab_size=len(vocab), emb_dim=cfg["model"]["emb_dim"],
                           hidden_dim=cfg["model"]["hidden_dim"], num_layers=cfg["model"]["num_layers"],
                           dr_rate=cfg["model"]["dr_rate"], fr_len=cfg["model"]["fr_len"]).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    # Log
    mlflow.set_tracking_uri("file:/" + hydra.utils.get_original_cwd() + "/mlruns")
    mlflow.start_run()
    mlflow.log_param("batch_size", cfg["train"]["batch_size"])
    mlflow.log_param("lr", cfg["train"]["lr"])
    mlflow.log_param("epoch", cfg["train"]["epoch"])

    train_loss = []
    test_loss = []
    for epoch in range(1+cfg["train"]["start_epoch"], cfg["train"]["start_epoch"]+cfg["train"]["epoch"]+1):
        train_loss_ep = []
        for i, (X, X_len) in enumerate(train_loader):
            optimizer.zero_grad()

            y = model(X.to(device), X_len)

            source = y[:, :-1, :]
            source = source.contiguous().view(-1, len(vocab))
            target = X[:, 1:]
            target = target.contiguous().view(-1)

            loss = criterion(source, target.to(device))
            loss.backward()
            optimizer.step()

            train_loss_ep.append(float(loss))
            # print("EPOCH%d:%d, Train loss:%f" % (epoch, i, loss))
        train_loss.append(np.mean(train_loss_ep))

        with torch.no_grad():
            test_loss_ep = []
            for j, (X, X_len) in enumerate(valid_loader):
                pred = model(X.to(device), X_len)

                source = pred[:, :-1, :]
                source = source.permute(0, 2, 1)
                target = X[:, 1:]

                loss = criterion(source, target.to(device))

                test_loss_ep.append(float(loss))

            test_loss.append(np.mean(test_loss_ep))

        print("EPOCH%d:, Train / Validation loss:%f / %f" %
              (epoch, float(np.mean(train_loss_ep)), float(np.mean(test_loss_ep))))

        if epoch % cfg["train"]["save_step"] == 0:
            torch.save(model.state_dict(),
                       hydra.utils.get_original_cwd() + cfg["train"]["ckptdir"] + f"model-ep{epoch}.pth")

        # Log
        mlflow.log_metric("Train total loss", float(np.mean(train_loss)), step=epoch)
        mlflow.log_metric("Test total loss", float(np.mean(test_loss)), step=epoch)

    with open(hydra.utils.get_original_cwd()+cfg["train"]["log_dir"]+cfg["train"]["log_filename"], "w") as f:
        f.write(cfg["train"]["log_filename"])
    mlflow.log_artifact(hydra.utils.get_original_cwd()+cfg["train"]["log_dir"]+cfg["train"]["log_filename"])
    mlflow.end_run()


@hydra.main(config_path="../config/", config_name="config")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    main()

