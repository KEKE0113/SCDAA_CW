import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class FFN(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, batch_norm=False):
        super().__init__()
        layers = [nn.BatchNorm1d(sizes[0]), ] if batch_norm else []
        for j in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[j], sizes[j + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(sizes[j + 1], affine=True))
            if j < (len(sizes) - 2):
                layers.append(activation())
            else:
                layers.append(output_activation())

        self.net = nn.Sequential(*layers)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        return self.net(x)


def build_control_network(input_dim=3, hidden_size=100, output_dim=2):
    """
    Exercise 2.2 / 4.1 shared control-network architecture.
    """
    return FFN(sizes=[input_dim, hidden_size, hidden_size, output_dim], activation=nn.Tanh)


def train_control_network(lqr, n_epochs=5000, batch_size=1000, lr=1e-3, x_range=3.0, net=None):
    """
    Supervised learning for the optimal Markov control.
    """
    net = build_control_network() if net is None else net
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        t_sample = torch.rand(batch_size, 1) * lqr.T
        x_sample = torch.empty(batch_size, 2).uniform_(-x_range, x_range)
        tx = torch.cat([t_sample, x_sample], dim=1)

        with torch.no_grad():
            a_label = lqr.control(t_sample.squeeze(1), x_sample.unsqueeze(1))

        a_pred = net(tx)
        loss = loss_fn(a_pred, a_label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % max(1, n_epochs // 10) == 0:
            print(f"Epoch {epoch}/{n_epochs}, loss={loss.item():.4f}")

    # plot traning loss
    plt.figure(figsize=(8, 5))
    plt.semilogy(losses)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Ex 2.2: Supervised Learning of Markov Control a(t,x)')
    plt.grid(True)
    plt.show()

    return net, losses

