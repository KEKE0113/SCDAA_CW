import numpy as np
import torch
from scipy.integrate import solve_ivp
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class LQR:
    def __init__(self, H, M, C, D, R, sigma, T):
        """
        Initialise the LQR problem with the given matrices and time horizon.
        """
        self.H = np.array(H, dtype=float)
        self.M = np.array(M, dtype=float)
        self.C = np.array(C, dtype=float)
        self.D = np.array(D, dtype=float)
        self.R = np.array(R, dtype=float)
        self.sigma = np.array(sigma, dtype=float)
        self.T = T

        self.dim_x = self.H.shape[0]
        self.dim_a = self.M.shape[1]
        self.D_inv = np.linalg.inv(self.D)
        self._riccati_ready = False


    @staticmethod
    def _to_numpy(x):
        """
        Convert input to a numpy array.
        """
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)


    def rhs(self, r, s_flat):
        """
        Right-hand side of the Riccati ODE, for use with scipy's solve_ivp.
        """
        S = s_flat.reshape(self.dim_x, self.dim_x)
        dS = -(self.H.T @ S + S @ self.H) + S @ self.M @ self.D_inv @ self.M.T @ S - self.C
        dS = 0.5 * (dS + dS.T)
        return dS.reshape(-1)


    def Sol_Ricatti(self, time_grid, rtol=1e-9, atol=1e-9):
        """
        Solve the Riccati ODE backwards in time from S(T) = R and store the interpolants
        for S(t) and the integral term.
        """
        time_grid = np.asarray(self._to_numpy(time_grid), dtype=float).reshape(-1)
        time_grid = np.unique(np.clip(time_grid, 0.0, self.T))
        if time_grid.size == 0:
            raise ValueError("time_grid must contain at least one point.")

        sol = solve_ivp(
            fun=self.rhs,
            t_span=(self.T, float(time_grid.min())),
            y0=self.R.reshape(-1),
            t_eval=np.sort(time_grid)[::-1],
            rtol=rtol,
            atol=atol,
        )
        if not sol.success:
            raise RuntimeError(f"Riccati ODE solver failed: {sol.message}")

        self.sol_t = sol.t
        self.sol_s = sol.y

        time_points = self.sol_t[::-1]
        S_ordered = self.sol_s[:, ::-1]
        S_mats = S_ordered.T.reshape(-1, self.dim_x, self.dim_x)
        S_mats = 0.5 * (S_mats + np.transpose(S_mats, (0, 2, 1)))

        self._interp_S = interp1d(
            time_points,
            S_mats,
            axis=0,
            bounds_error=False,
            fill_value="extrapolate",
        )

        sig_sig_T = self.sigma @ self.sigma.T
        traces = np.einsum("ij,tji->t", sig_sig_T, S_mats)
        integral_0_to_t = cumulative_trapezoid(traces, time_points, initial=0.0)
        integral_t_to_T = integral_0_to_t[-1] - integral_0_to_t
        self._interp_integral = interp1d(
            time_points,
            integral_t_to_T,
            bounds_error=False,
            fill_value="extrapolate",
        )
        self._riccati_ready = True


    def _ensure_riccati(self):
        """
        Ensure the Riccati ODE has been solved before any query method is called.
        If Sol_Ricatti has not yet been called, automatically solves on a dense default
        grid of 2000 points.
        """
        if not self._riccati_ready:
            dense_grid = np.linspace(0.0, self.T, 2000)
            self.Sol_Ricatti(dense_grid)


    def S_of_t(self, t):
        """
        Return the Riccati solution S(t) at a batch of time points using the precomputed
        interpolant.
        """
        self._ensure_riccati()
        t_np = np.asarray(self._to_numpy(t), dtype=float).reshape(-1)
        return np.asarray(self._interp_S(t_np))


    def Sol_value(self, t, x):
        """
        Compute the LQR value function v(t, x) for a batch of (t, x) pairs.
        """
        if isinstance(x, torch.Tensor) and x.dim() == 3:
            x = x.squeeze(1)
        elif isinstance(x, np.ndarray) and x.ndim == 3:
            x = x.squeeze(1)

        self._ensure_riccati()
        t_np = np.asarray(self._to_numpy(t), dtype=float).reshape(-1)
        x_np = np.asarray(self._to_numpy(x), dtype=float).reshape(-1, self.dim_x)
        if t_np.shape[0] != x_np.shape[0]:
            raise ValueError("t and x must have the same batch size.")

        S_batch = self.S_of_t(t_np)
        xSx = np.einsum("bi,bij,bj->b", x_np, S_batch, x_np)
        integral_batch = np.asarray(self._interp_integral(t_np))
        values = (xSx + integral_batch).reshape(-1, 1)
        return torch.tensor(values, dtype=torch.float32)


    def control(self, t, x):
        """
        Compute the optimal Markov control a(t, x) for a batch of (t, x) pairs.
        """
        if isinstance(x, torch.Tensor) and x.dim() == 3:
            x = x.squeeze(1)
        elif isinstance(x, np.ndarray) and x.ndim == 3:
            x = x.squeeze(1)

        self._ensure_riccati()
        t_np = np.asarray(self._to_numpy(t), dtype=float).reshape(-1)
        x_np = np.asarray(self._to_numpy(x), dtype=float).reshape(-1, self.dim_x)
        if t_np.shape[0] != x_np.shape[0]:
            raise ValueError("t and x must have the same batch size.")

        S_batch = self.S_of_t(t_np)
        controls = -np.einsum("ab,tbc,tc->ta", self.D_inv @ self.M.T, S_batch, x_np)
        return torch.tensor(controls, dtype=torch.float32)

    def monte_carlo_with_control(self, t, x, N_steps, N_samples, control_fn, seed=None):
        """
        Generic Euler Monte Carlo routine used by Exercises 1.2 and 3.1.
        """
        t = float(t)
        x = np.asarray(x, dtype=float).reshape(self.dim_x)
        N_steps = int(N_steps)
        N_samples = int(N_samples)
        if N_steps <= 0 or N_samples <= 0:
            raise ValueError("N_steps and N_samples must be positive integers.")

        rng = np.random.default_rng(seed)
        dt = (self.T - t) / N_steps
        time_grid = np.linspace(t, self.T, N_steps + 1)

        X = np.tile(x, (N_samples, 1))
        cost = np.zeros(N_samples)

        for n in range(N_steps):
            tn = time_grid[n]
            a = np.asarray(control_fn(tn, X), dtype=float).reshape(N_samples, self.dim_a)

            running_cost = np.einsum("bi,ij,bj->b", X, self.C, X) + np.einsum("bi,ij,bj->b", a, self.D, a)
            cost += running_cost * dt

            dW = rng.standard_normal((N_samples, self.dim_x)) * np.sqrt(dt)
            X = X + dt * (X @ self.H.T + a @ self.M.T) + dW @ self.sigma.T

        terminal_cost = np.einsum("bi,ij,bj->b", X, self.R, X)
        cost += terminal_cost
        return float(np.mean(cost))

    def monte_carlo(self, t, x, N_steps, N_samples, seed=None):
        """
        Euler MC for the optimally controlled dynamics.
        """
        def optimal_control_fn(time, X_batch):
            controls = self.control(
                torch.full((X_batch.shape[0],), time, dtype=torch.float32),
                torch.tensor(X_batch[:, None, :], dtype=torch.float32),
            )
            return controls.detach().cpu().numpy()

        return self.monte_carlo_with_control(t, x, N_steps, N_samples, optimal_control_fn, seed=seed)

    def monte_carlo_constant_control(self, t, x, N_steps, N_samples, alpha, seed=None):
        """
        adapt the MC routine to a fixed control alpha.
        """
        alpha = np.asarray(alpha, dtype=float).reshape(self.dim_a)

        def constant_control_fn(_time, X_batch):
            return np.tile(alpha, (X_batch.shape[0], 1))

        return self.monte_carlo_with_control(t, x, N_steps, N_samples, constant_control_fn, seed=seed)

    def error_analysis(self, t, x, steps_list=None, samples_list=None,
            fixed_paths=100000, fixed_steps=5000, repetitions=3, seed=1234, show=True):
        """
        Produce two log-log convergence plots to validate the Monte Carlo estimator against
        the exact value function from the Riccati solution.
        """
        self._ensure_riccati()
        x = np.asarray(x, dtype=float).reshape(self.dim_x)
        if steps_list is None:
            steps_list = [1, 10, 50, 100, 500, 1000, 5000]
        if samples_list is None:
            samples_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]

        v_exact = self.Sol_value(
            torch.tensor([t], dtype=torch.float32),
            torch.tensor(x[None, :], dtype=torch.float32),
        ).item()
        print(f"Benchmark value v(t,x) = {v_exact:.6f}")

        def averaged_error(n_steps, n_samples, offset):
            errs = []
            for k in range(repetitions):
                v_mc = self.monte_carlo(t, x, n_steps, n_samples, seed=seed + offset + k)
                errs.append(abs(v_mc - v_exact))
            return float(np.mean(errs))

        errors_steps = []
        for i, n_steps in enumerate(steps_list):
            err = averaged_error(n_steps, fixed_paths, 1000 * i)
            errors_steps.append(err)
            print(f"N_steps={n_steps:5d}, mean abs error={err:.4e}")

        errors_samples = []
        for i, n_samples in enumerate(samples_list):
            err = averaged_error(fixed_steps, n_samples, 100000 + 1000 * i)
            errors_samples.append(err)
            print(f"N_samples={n_samples:6d}, mean abs error={err:.4e}")

        fig1 = plt.figure(figsize=(8, 5))
        plt.loglog(steps_list, errors_steps, "bo-", label="MC error")
        ref_step = errors_steps[0] * (steps_list[0] / np.asarray(steps_list))
        plt.loglog(steps_list, ref_step, "r--", label=r"$O(1/N)$ reference")
        plt.xlabel("Number of time steps N")
        plt.ylabel(r"Absolute error $|v_{MC}-v|$")
        plt.title("Exercise 1.2: Error vs Time Steps")
        plt.legend()
        plt.grid(True, which="both")

        fig2 = plt.figure(figsize=(8, 5))
        plt.loglog(samples_list, errors_samples, "bo-", label="MC error")
        ref_sample = errors_samples[0] * np.sqrt(samples_list[0] / np.asarray(samples_list))
        plt.loglog(samples_list, ref_sample, "r--", label=r"$O(M^{-1/2})$ reference")
        plt.xlabel("Number of Monte Carlo paths M")
        plt.ylabel(r"Absolute error $|v_{MC}-v|$")
        plt.title(f"Exercise 1.2: Error vs MC Samples (N={fixed_steps})")
        plt.legend()
        plt.grid(True, which="both")

        if show:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)

        return {
            "v_exact": v_exact,
            "steps_list": steps_list,
            "errors_steps": errors_steps,
            "samples_list": samples_list,
            "errors_samples": errors_samples,
        }



