import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class DGM_Layer(nn.Module):
    def __init__(self, dim_x, dim_S):
        super(DGM_Layer, self).__init__()
        self.activation = nn.Tanh()
        self.gate_Z = self.layer(dim_x + dim_S, dim_S)
        self.gate_G = self.layer(dim_x + dim_S, dim_S)
        self.gate_R = self.layer(dim_x + dim_S, dim_S)
        self.gate_H = self.layer(dim_x + dim_S, dim_S)

    def layer(self, nIn, nOut):
        return nn.Sequential(nn.Linear(nIn, nOut), self.activation)

    def forward(self, x, S):
        x_S = torch.cat([x, S], 1)
        Z = self.gate_Z(x_S)
        G = self.gate_G(x_S)
        R = self.gate_R(x_S)
        H = self.gate_H(torch.cat([x, S * R], 1))
        return (1 - G) * H + Z * S

class Net_DGM(nn.Module):
    def __init__(self, dim_x, dim_S):
        super(Net_DGM, self).__init__()
        self.input_layer = nn.Sequential(nn.Linear(dim_x+1, dim_S), nn.Tanh())
        self.DGM1 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S)
        self.DGM2 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S)
        self.DGM3 = DGM_Layer(dim_x=dim_x+1, dim_S=dim_S)
        self.output_layer = nn.Linear(dim_S, 1)

    def forward(self,t,x):
        tx = torch.cat([t,x], 1)
        S1 = self.input_layer(tx)
        S2 = self.DGM1(tx,S1)
        S3 = self.DGM2(tx,S2)
        S4 = self.DGM3(tx,S3)
        return self.output_layer(S4)


def build_value_network(dim_x=2, hidden_size=100):
    """Exercise 2.1 / 4.1 shared value-network architecture."""
    return Net_DGM(dim_x=dim_x, dim_S=hidden_size)


def train_value_network(lqr, n_epochs=5000, batch_size=1000, lr=1e-3, x_range=3.0, net=None):
    """Exercise 2.1: supervised learning for the value function."""
    net = build_value_network(dim_x=2, hidden_size=100) if net is None else net
    optimizer = optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    losses = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        t_sample = torch.rand(batch_size, 1) * lqr.T
        x_sample = torch.empty(batch_size, 2).uniform_(-x_range, x_range)

        with torch.no_grad():
            v_label = lqr.Sol_value(t_sample.squeeze(1), x_sample.unsqueeze(1))

        v_pred = net(t_sample, x_sample)
        loss = loss_fn(v_pred, v_label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if epoch % max(1, n_epochs // 10) == 0:
            print(f"Epoch {epoch}/{n_epochs}, loss={loss.item():.6f}")

    plt.figure(figsize=(8, 5))
    plt.semilogy(losses)
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Exercise 2.1: Supervised Learning of Value Function")
    plt.grid(True)
    plt.show()

    return net, losses