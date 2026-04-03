import torch
import numpy as np
import matplotlib.pyplot as plt
from DGM import *


def quadratic_form_batch(x, A):
    return (x @ A * x).sum(dim=1, keepdim=True)


def diffusion_trace_term(u_x, x, sig_sig_T):
    """Compute tr((sigma sigma^T) Hessian(u)) in 2D."""
    batch_size, dim_x = x.shape
    hess_cols = []
    for i in range(dim_x):
        grad_i = torch.autograd.grad(
            u_x[:, i:i + 1],
            x,
            grad_outputs=torch.ones_like(u_x[:, i:i + 1]),
            create_graph=True,
            retain_graph=True,
        )[0]
        hess_cols.append(grad_i.unsqueeze(2))
    hess = torch.cat(hess_cols, dim=2)  # [N, dim_x, dim_x]
    return torch.einsum("ij,nji->n", sig_sig_T, hess).view(batch_size, 1)


def build_default_dgm_eval_points(T):
    """
    Fixed validation set used throughout Exercise 3.1 so that the error curve is
    comparable across training epochs.
    """
    times = [0.0, 0.25 * T, 0.50 * T]
    states = [
        np.array([1.0, 1.0], dtype=float),
        np.array([0.0, 1.0], dtype=float),
        np.array([1.0, -1.0], dtype=float),
    ]
    return [(float(t0), x0.copy()) for t0 in times for x0 in states]


def precompute_dgm_mc_benchmark(
    lqr,
    eval_points,
    N_steps=500,
    N_samples=5000,
    alpha=None,
    base_seed=2026,
):
    """
    Reuse the Exercise 1.2 Monte Carlo routine, with the optimal control replaced
    by the constant control alpha=(1,1)^T, and precompute the benchmark values once.
    """
    if alpha is None:
        alpha = np.array([1.0, 1.0], dtype=float)
    else:
        alpha = np.asarray(alpha, dtype=float).reshape(-1)

    mc_values = []
    for idx, (t0, x0) in enumerate(eval_points):
        value = lqr.monte_carlo_constant_control(
            t=t0,
            x=np.asarray(x0, dtype=float),
            N_steps=N_steps,
            N_samples=N_samples,
            alpha=alpha,
            seed=base_seed + idx,
        )
        mc_values.append(float(value))
    return np.asarray(mc_values, dtype=float)


@torch.no_grad()
def evaluate_dgm_errors(net, eval_points, mc_values):
    """
    Compute the mean absolute error against a fixed Monte Carlo benchmark.
    """
    t_eval = torch.tensor([[pt[0]] for pt in eval_points], dtype=torch.float32)
    x_eval = torch.tensor(np.stack([pt[1] for pt in eval_points]), dtype=torch.float32)
    u_pred = net(t_eval, x_eval).squeeze(1).cpu().numpy()
    return float(np.mean(np.abs(u_pred - mc_values)))


def train_dgm_linear_pde(
    lqr,
    n_epochs=5000,
    batch_size=256,
    lr=1e-3,
    eval_every=500,
    x_range=3.0,
    net=None,
    eval_points=None,
    benchmark_steps=500,
    benchmark_samples=5000,
):
    """Exercise 3.1: DGM for the linear PDE under constant control alpha=(1,1)."""
    alpha = torch.tensor([1.0, 1.0], dtype=torch.float32)
    T = lqr.T

    H = torch.tensor(lqr.H, dtype=torch.float32)
    M = torch.tensor(lqr.M, dtype=torch.float32)
    C = torch.tensor(lqr.C, dtype=torch.float32)
    D = torch.tensor(lqr.D, dtype=torch.float32)
    R = torch.tensor(lqr.R, dtype=torch.float32)
    sigma = torch.tensor(lqr.sigma, dtype=torch.float32)
    sig_sig_T = sigma @ sigma.T

    alpha_D_alpha = float(alpha @ D @ alpha)
    Malpha = M @ alpha

    net = build_value_network(dim_x=2, hidden_size=100) if net is None else net
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=max(1, n_epochs // 3),
        gamma=0.5,
    )

    if eval_points is None:
        eval_points = build_default_dgm_eval_points(T)
    mc_values = precompute_dgm_mc_benchmark(
        lqr,
        eval_points=eval_points,
        N_steps=benchmark_steps,
        N_samples=benchmark_samples,
        alpha=alpha.detach().cpu().numpy(),
    )

    losses, eqn_losses, bdy_losses = [], [], []
    eval_epochs, mc_errors = [], []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        t_int = torch.rand(batch_size, 1) * T
        t_int.requires_grad_(True)
        x_int = torch.empty(batch_size, 2).uniform_(-x_range, x_range)
        x_int.requires_grad_(True)

        u = net(t_int, x_int)
        u_t = torch.autograd.grad(
            u,
            t_int,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        u_x = torch.autograd.grad(
            u,
            x_int,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        trace_term = diffusion_trace_term(u_x, x_int, sig_sig_T)
        term_Hx = (u_x * (x_int @ H.T)).sum(dim=1, keepdim=True)
        term_Malpha = (u_x * Malpha.view(1, -1)).sum(dim=1, keepdim=True)
        xCx = quadratic_form_batch(x_int, C)

        pde_residual = u_t + 0.5 * trace_term + term_Hx + term_Malpha + xCx + alpha_D_alpha
        loss_eqn = (pde_residual ** 2).mean()

        x_bdy = torch.empty(batch_size, 2).uniform_(-x_range, x_range)
        t_bdy = torch.full((batch_size, 1), T)
        u_bdy = net(t_bdy, x_bdy)
        xRx = quadratic_form_batch(x_bdy, R)
        loss_bdy = ((u_bdy - xRx) ** 2).mean()

        loss = loss_eqn + loss_bdy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        eqn_losses.append(loss_eqn.item())
        bdy_losses.append(loss_bdy.item())

        if epoch % max(1, n_epochs // 10) == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(
                f"Epoch {epoch}/{n_epochs} | total={loss.item():.6f} | "
                f"pde={loss_eqn.item():.6f} | boundary={loss_bdy.item():.6f} | lr={current_lr:.2e}"
            )

        if epoch % eval_every == 0:
            err = evaluate_dgm_errors(net, eval_points, mc_values)
            eval_epochs.append(epoch)
            mc_errors.append(err)
            print(f"  -> mean MC benchmark error on fixed validation set = {err:.4e}")

    plt.figure(figsize=(8, 5))
    plt.semilogy(losses, label="Total loss")
    plt.semilogy(eqn_losses, label="PDE residual loss")
    plt.semilogy(bdy_losses, label="Terminal loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Exercise 3.1: DGM Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.semilogy(eval_epochs, mc_errors, "bo-")
    plt.xlabel("Epoch")
    plt.ylabel("Mean absolute error vs MC")
    plt.title("Exercise 3.1: DGM Error vs Monte Carlo Benchmark")
    plt.grid(True)
    plt.show()

    return net, {
        "losses": losses,
        "eqn_losses": eqn_losses,
        "bdy_losses": bdy_losses,
        "eval_epochs": eval_epochs,
        "mc_errors": mc_errors,
        "eval_points": eval_points,
        "mc_values": mc_values.tolist(),
    }
