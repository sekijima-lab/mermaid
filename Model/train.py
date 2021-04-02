import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Model.dataset import MolDataset
from Model.model import RolloutNetwork
from Utils.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(num_epoch, smiles_list, vocab, seq_len, batch_size, log_dir, ckpt_dir, lr, test_ratio):
    smiles_list = [convert_smiles(parse_smiles("&" + s.rstrip("\n") + "\n"), vocab, mode="s2i") for s in smiles_list]

    sp = int(len(smiles_list) * (1 - test_ratio))
    train_dataset = MolDataset(smiles_list[:sp], seq_len)
    valid_dataset = MolDataset(smiles_list[sp:], seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model = RolloutNetwork(vocab_size=len(vocab))
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    writer = SummaryWriter(log_dir=log_dir)
    train_loss = []
    test_loss = []
    for epoch in range(1, num_epoch+1):
        train_loss_ep = []
        for i, (X, X_len) in enumerate(train_loader):
            optimizer.zero_grad()

            y = model(X.cuda(), X_len)

            source = y[:, :-1, :]
            source = source.contiguous().view(-1, len(vocab))
            target = X[:, 1:]
            target = target.contiguous().view(-1)

            loss = criterion(source, target.cuda())
            loss.backward()
            optimizer.step()

            train_loss_ep.append(float(loss))
            print("EPOCH%d:%d, Train loss:%f" % (epoch, i, loss))
        train_loss.append(np.mean(train_loss_ep))

        with torch.no_grad():
            test_loss_ep = []
            for j, (X, X_len) in enumerate(valid_loader):
                pred = model(X.cuda(), X_len)

                source = pred[:, :-1, :]
                source = source.permute(0, 2, 1)
                target = X[:, 1:]

                loss = criterion(source, target.cuda())

                test_loss_ep.append(float(loss))

            test_loss.append(np.mean(test_loss_ep))
            print("EPOCH%d:, Validation loss:%f" % (epoch, np.mean(test_loss_ep)))

        torch.save(model.state_dict(), f"{ckpt_dir}/ckpt-{epoch}.pth")
        writer.add_scalar("train_loss", train_loss[-1], epoch)
        writer.add_scalar("test_loss", test_loss[-1], epoch)

    sns.set()
    sns.set_style('ticks')
    fig, ax = plt.subplots(figsize=(10, 10))
    ind = [i for i in range(len(train_loss))]
    sns.lineplot(ind, train_loss)
    sns.lineplot(ind, test_loss)
    sns.despine()
    ax.set(
        xlabel="Epoch",
        ylabel="Negative Log Likelihood",
    )
    ax.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seq_len", type=int, default=25)
    parser.add_argument("--datapath", type=str, default="Data/250k_rndm_zinc_drugs_clean.smi")
    parser.add_argument("--logdir", type=str, default="logs")
    parser.add_argument("--ckptdir", type=str, default="checkpoint")
    args = parser.parse_args()

    fragment_list = read_smilesset(args.datapath)

    vocab = VOCABULARY
    train(num_epoch=args.epoch,
          smiles_list=fragment_list,
          vocab=vocab,
          seq_len=args.seq_len,
          batch_size=args.batch_size,
          log_dir=args.logdir,
          ckpt_dir=args.ckptdir,
          lr=args.lr,
          test_ratio=args.test_ratio,
          )

