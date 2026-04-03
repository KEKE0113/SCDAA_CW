import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from DGM import build_value_network
from FNN import build_control_network

def quadratic_form_batch(x, A):
    return (x @ A * x).sum(dim=1, keepdim=True)


def diffusion_trace_term(u_x, x, sig_sig_T):
    batch_size, dim_x = x.shape
    hess_cols = []
    for i in range(dim_x):
        grad_i = torch.autograd.grad(
            u_x[:, i:i+1],
            x,
            grad_outputs=torch.ones_like(u_x[:, i:i+1]),
            create_graph=True,
            retain_graph=True,
        )[0]
        hess_cols.append(grad_i.unsqueeze(2))
    hess = torch.cat(hess_cols, dim=2)
    return torch.einsum("ij,nji->n", sig_sig_T, hess).view(batch_size, 1)


@torch.no_grad()
def evaluate_policy_iteration(net_val, net_act, lqr, n_test=128, x_range=2.0):
    t_test = torch.rand(n_test, 1) * lqr.T
    x_test = torch.empty(n_test, 2).uniform_(-x_range, x_range)

    v_exact = lqr.Sol_value(t_test.squeeze(1), x_test).squeeze(1).numpy()
    v_pred = net_val(t_test, x_test).squeeze(1).numpy()
    err_val = float(np.mean(np.abs(v_pred - v_exact)))

    a_exact = lqr.control(t_test.squeeze(1), x_test).numpy()
    a_pred = net_act(torch.cat([t_test, x_test], dim=1)).numpy()
    err_act = float(np.mean(np.linalg.norm(a_pred - a_exact, axis=1)))

    return err_val, err_act


def train_policy_iteration(
    lqr,
    n_iterations=10,
    n_epochs_val=2000,
    n_epochs_act=1000,
    batch_size=256,
    lr=1e-3,
    x_range=3.0,
    initial_value_net=None,
    initial_control_net=None,
):
    """Exercise 4.1: policy iteration with one value net and one control net."""
    H = torch.tensor(lqr.H, dtype=torch.float32)
    M = torch.tensor(lqr.M, dtype=torch.float32)
    C = torch.tensor(lqr.C, dtype=torch.float32)
    D = torch.tensor(lqr.D, dtype=torch.float32)
    R = torch.tensor(lqr.R, dtype=torch.float32)
    sigma = torch.tensor(lqr.sigma, dtype=torch.float32)
    sig_sig_T = sigma @ sigma.T
    T = lqr.T

    net_val = copy.deepcopy(initial_value_net) if initial_value_net is not None else build_value_network(dim_x=2, hidden_size=100)
    net_act = copy.deepcopy(initial_control_net) if initial_control_net is not None else build_control_network()

    if initial_value_net is not None:
        print("Using Exercise 2.1 value network as the initial guess for policy iteration.")
    if initial_control_net is not None:
        print("Using Exercise 2.2 control network as the initial guess for policy iteration.")

    iter_errors_val, iter_errors_act = [], []

    for iteration in range(n_iterations):
        print("\n" + "=" * 60)
        print(f"Policy iteration {iteration + 1}/{n_iterations}")
        print("=" * 60)

        optimizer_val = torch.optim.Adam(net_val.parameters(), lr=lr)
        for epoch in range(n_epochs_val):
            optimizer_val.zero_grad()

            t = torch.rand(batch_size, 1) * T
            t.requires_grad_(True)
            x = torch.empty(batch_size, 2).uniform_(-x_range, x_range)
            x.requires_grad_(True)

            tx = torch.cat([t.detach(), x.detach()], dim=1)
            a = net_act(tx)
            u = net_val(t, x)

            u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]

            trace_term = diffusion_trace_term(u_x, x, sig_sig_T)
            term_Hx = (u_x * (x @ H.T)).sum(dim=1, keepdim=True)
            term_Ma = (u_x * (a @ M.T)).sum(dim=1, keepdim=True)
            term_xCx = quadratic_form_batch(x, C)
            term_aDa = quadratic_form_batch(a, D)

            pde_residual = u_t + 0.5 * trace_term + term_Hx + term_Ma + term_xCx + term_aDa
            loss_pde = (pde_residual ** 2).mean()

            x_T = torch.empty(batch_size, 2).uniform_(-x_range, x_range)
            t_T = torch.full((batch_size, 1), T)
            u_T = net_val(t_T, x_T)
            xRx = quadratic_form_batch(x_T, R)
            loss_bdy = ((u_T - xRx) ** 2).mean()

            loss = loss_pde + loss_bdy
            loss.backward()
            optimizer_val.step()

            if epoch % max(1, n_epochs_val // 5) == 0:
                print(f"  [Value] epoch {epoch}/{n_epochs_val}, loss={loss.item():.6f}")

        optimizer_act = torch.optim.Adam(net_act.parameters(), lr=lr)
        for epoch in range(n_epochs_act):
            optimizer_act.zero_grad()

            t = torch.rand(batch_size, 1) * T
            x = torch.empty(batch_size, 2).uniform_(-x_range, x_range)
            x.requires_grad_(True)

            u = net_val(t, x)
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=False, retain_graph=False)[0].detach()

            a = net_act(torch.cat([t, x.detach()], dim=1))
            Hx = x.detach() @ H.T
            Ma = a @ M.T

            Hamiltonian = (u_x * Hx).sum(dim=1, keepdim=True) + (u_x * Ma).sum(dim=1, keepdim=True)
            Hamiltonian += quadratic_form_batch(x.detach(), C) + quadratic_form_batch(a, D)
            loss_act = Hamiltonian.mean()  # minimise over controls
            loss_act.backward()
            optimizer_act.step()

            if epoch % max(1, n_epochs_act // 5) == 0:
                print(f"  [Control] epoch {epoch}/{n_epochs_act}, Hamiltonian={loss_act.item():.6f}")

        err_val, err_act = evaluate_policy_iteration(net_val, net_act, lqr)
        iter_errors_val.append(err_val)
        iter_errors_act.append(err_act)
        print(f"  mean |v_pred-v_exact| = {err_val:.4e}")
        print(f"  mean ||a_pred-a_exact|| = {err_act:.4e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.semilogy(range(1, n_iterations + 1), iter_errors_val, "bo-")
    ax1.set_xlabel("Policy iteration")
    ax1.set_ylabel("Mean absolute value error")
    ax1.set_title("Exercise 4.1: Value Error per Iteration")
    ax1.grid(True)

    ax2.semilogy(range(1, n_iterations + 1), iter_errors_act, "ro-")
    ax2.set_xlabel("Policy iteration")
    ax2.set_ylabel("Mean control error")
    ax2.set_title("Exercise 4.1: Control Error per Iteration")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()
    return net_val, net_act, iter_errors_val, iter_errors_act
