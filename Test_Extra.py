import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from DGM import build_value_network


def plot_train_val_loss_original(lqr):
    np.random.seed(2026)
    torch.manual_seed(2026)

    net = build_value_network()

    n_epochs = 1500
    batch_size = 1024
    lr = 1e-3
    x_range = 3.0

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        net.train()
        optimizer.zero_grad()

        t_train = torch.rand(batch_size, 1) * lqr.T
        x_train = torch.empty(batch_size, 2).uniform_(-x_range, x_range)
        with torch.no_grad():
            v_train_exact = lqr.Sol_value(t_train.squeeze(1), x_train.unsqueeze(1))

        v_train_pred = net(t_train, x_train)
        loss = loss_fn(v_train_pred, v_train_exact)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                t_val = torch.rand(batch_size, 1) * lqr.T
                x_val = torch.empty(batch_size, 2).uniform_(-x_range, x_range)
                v_val_exact = lqr.Sol_value(t_val.squeeze(1), x_val.unsqueeze(1))
                v_val_pred = net(t_val, x_val)
                val_loss = loss_fn(v_val_pred, v_val_exact)

            train_losses.append(loss.item())
            val_losses.append(val_loss.item())

            if epoch % 150 == 0:
                print(f"Epoch {epoch:4d}/{n_epochs} | Train Loss: {loss.item():.4e} | Val Loss: {val_loss.item():.4e}")

    plt.figure(figsize=(8, 5))
    plt.semilogy(range(0, n_epochs, 10), train_losses, 'b-', label="Training Loss")
    plt.semilogy(range(0, n_epochs, 10), val_losses, 'r--', label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Test: Training Lossvs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()