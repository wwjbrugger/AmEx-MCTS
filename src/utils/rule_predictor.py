import torch
from torch.optim import Adam
from torch import save, load, nn
import torch.nn.functional as F


class RulePredictor:
    def __init__(self, net, args):
        self.net = net
        self.optimizer = Adam(self.net.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
        self.env_steps = 0

    def predict(self, state):
        return self.net(state)

    def numpy(self, x):
        with torch.no_grad():
            pi, v = self.predict([x])

        return pi[0].numpy(), v[0].numpy()

    def predict_with_loss(self, batch):
        states, pi, z = batch
        p, v = self.predict(states)
        value_loss = F.mse_loss(torch.tensor(z), v)
        pi_loss = -F.cross_entropy(torch.tensor(pi), p)

        loss = value_loss + pi_loss
        return v, p, loss, value_loss.item(), pi_loss.item()

    def train(self, batch):
        self.net.train(True)

        self.optimizer.zero_grad()
        p, v, loss, value_loss, pi_loss = self.predict_with_loss(batch)
        loss.backward()

        self.optimizer.step()

        self.net.train(False)
        return value_loss, pi_loss

    def load_checkpoint(self, path) -> int:
        checkpoint = load(path)
        self.net = checkpoint["net_cls"]()
        self.net.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.env_steps = checkpoint["env_steps"]
        return checkpoint["epoch"]

    def save_checkpoint(self, epoch, path):
        save({
            "epoch": epoch,
            "net_cls": self.net.__class__,
            "model_state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "env_steps": self.env_steps,
        }, path)


class NN(nn.Module):
    def __init__(self, game):
        super().__init__()
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, game.getActionSize() + 1)

    def forward(self, x):
        obs_values = [entry['obs'] for entry in x]

        one_hot = torch.zeros(len(obs_values), 16)
        one_hot[:, obs_values] = 1
        x = F.relu(self.fc2(one_hot))
        x = self.fc3(x)
        return F.softmax(x[:, :-1], dim=1), x[:, -1]
