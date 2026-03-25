from __future__ import annotations
from functools import partial
import argparse

import jax
import jax.numpy as jnp

import hodel
from helper import get_triplets, TripletAux

jax.config.update("jax_enable_x64", True)

method_dict = {
    "hodel": hodel.Method.HODEL,
    "pinn": hodel.Method.PINN,
    "deq": hodel.Method.DEQ,
    "node": hodel.Method.ODE,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D rod benchmarks.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["hodel", "pinn", "deq", "node"],
        help="The learning framework to use.",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="Eta noise scaling factor (defaults: 0.0).",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=100,
        help="# of lambdas between [0, 1.0].",
    )
    parser.add_argument(
        "--n_disps",
        type=int,
        default=20,
        help="# of displacements per K.",
    )
    parser.add_argument(
        "--n_disps_hessian",
        type=int,
        default=100,
        help="# of displacements per K in hessian.",
    )
    args = parser.parse_args()
    method = method_dict[args.method]
    sim, state, xf0, K, get_aux = get_triplets()

    def get_aux_xf_stars(
        key: jax.Array,
        eta: float = 0.01,
        n_steps: int = 100,
    ) -> tuple[TripletAux, jax.Array]:
        # Split keys for fairness
        k_disp, k_theta, k_noise = jax.random.split(key, 3)
        disp_raw = jax.random.uniform(k_disp, (3,)) * 2 - 1
        disp = disp_raw * jnp.array([0.01, 0.015, 0.015])
        theta = jax.random.uniform(k_theta, (1,)) * 2 - 1
        # jax.debug.print("disp: {}, theta: {}", disp, theta)

        # Generate trajectory with truth and add noise
        lambdas = jnp.linspace(0, 1.0, n_steps)
        aux = get_aux(disp, theta)
        xf_stars = sim.solve(lambdas, xf0, K, aux, state)
        return aux, xf_stars * (1 + eta * jax.random.normal(k_noise, xf_stars.shape))

    # Get displacements
    lambdas = jnp.linspace(0, 1.0, args.n_steps)
    keys = jax.vmap(lambda i: jax.random.PRNGKey(i))(jnp.arange(args.n_disps_hessian))
    batch_aux, batch_xf_stars = jax.vmap(
        lambda key: get_aux_xf_stars(key, eta=args.eta, n_steps=args.n_steps)
    )(keys)

    def get_accumulated_hessian(K_: jax.Array, method: hodel.Method) -> jax.Array:
        """Get accumulated Gauss-newton Hessian."""
        num_sims = batch_xf_stars.shape[0]
        num_params = K_.size
        H_init = jnp.zeros((num_params, num_params))

        def body_fn(H_acc, i):
            aux_i = jax.tree_util.tree_map(lambda x: x[i], batch_aux)
            xf_stars_i = batch_xf_stars[i]

            def solve_residual(K_inner):
                pred = sim.solve(lambdas, xf0, K_inner, aux_i, state)
                res = (xf_stars_i - pred).flatten()
                scale = jnp.linalg.norm(pred) + 1e-8
                return res / scale

            def ode_residual(K_inner):
                pred = sim.ode_solve(lambdas, xf0, K_inner, aux_i, state)
                res = (xf_stars_i - pred).flatten()
                scale = jnp.linalg.norm(pred) + 1e-8
                return res / scale

            def res_residual(K_inner):
                xbs = jax.vmap(sim.get_xb, in_axes=(0, None))(lambdas, aux_i)
                Ws = jax.vmap(sim.get_W, in_axes=(0, None))(lambdas, aux_i)
                residual_fn = partial(
                    sim._residual_core, Theta=K_inner, aux=aux_i, carry=state
                )
                res = jax.vmap(residual_fn)(xf_stars_i, xbs, Ws).flatten()
                scale = jnp.linalg.norm(res) + 1e-8
                return res / scale

            def deq_residual(K_inner):
                xbs = jax.vmap(sim.get_xb, in_axes=(0, None))(lambdas, aux_i)
                Ws = jax.vmap(sim.get_W, in_axes=(0, None))(lambdas, aux_i)
                pred = jax.vmap(
                    lambda xb, W: sim.solve_fn(xf0, xb, W, K_inner, aux_i, state, 15)
                )(xbs, Ws)
                res = (xf_stars_i - pred).flatten()
                scale = jnp.linalg.norm(pred) + 1e-8
                return res / scale

            if method == hodel.Method.PINN:
                res_fn = res_residual
            elif method == hodel.Method.HODEL:
                res_fn = solve_residual
            elif method == hodel.Method.ODE:
                res_fn = ode_residual
            elif method == hodel.Method.DEQ:
                res_fn = deq_residual
            Ji = jax.jacobian(res_fn)(K_)
            H_acc += Ji.T @ Ji
            return H_acc, None

        total_H, _ = jax.lax.scan(body_fn, H_init, jnp.arange(num_sims))
        return total_H

    # Get average loss Hessian
    H = get_accumulated_hessian(K, method) / batch_xf_stars.shape[0]

    # Use root h for eigenvector for normality
    if method == hodel.Method.PINN:
        data = jnp.load("data/pinn_H.npz")
    else:
        data = jnp.load("data/root_H.npz")
    H_root = data["H"]
    evals = jnp.linalg.eigvalsh(H_root)
    cond_num = evals[-1] / evals[0]
    eigenvalues, eigenvectors = jnp.linalg.eigh(H)
    idx = jnp.argsort(eigenvalues)[::-1]
    v1 = eigenvectors[:, idx[0]]
    v2 = eigenvectors[:, idx[1]]

    # Get K_set
    n_points = 25
    steps = jnp.linspace(-0.2, 0.2, n_points)
    alpha, beta = jnp.meshgrid(steps, steps)
    K_grid = K * jnp.exp(alpha[..., None] * v1 + beta[..., None] * v2)  # parameters > 0
    K_flat = K_grid.reshape(-1, 4)

    # Get displacements for contour
    lambdas = jnp.linspace(0, 1.0, args.n_steps)
    keys = jax.vmap(lambda i: jax.random.PRNGKey(i))(jnp.arange(args.n_disps))
    batch_aux, batch_xf_stars = jax.vmap(
        lambda key: get_aux_xf_stars(key, eta=args.eta, n_steps=args.n_steps)
    )(keys)

    @jax.jit
    def get_loss_over_bcs(K_):
        losses = jax.vmap(
            lambda aux_, xf_stars_: sim.loss(
                lambdas, xf0, xf_stars_, K_, aux_, state, method=method
            )
        )(batch_aux, batch_xf_stars)
        # jax.debug.print("{} {}", jnp.nanmedian(losses), jnp.nanvar(losses))
        return losses

    Ls = jax.lax.map(get_loss_over_bcs, K_flat)
    save_path = f"results_{args.method}_eta={args.eta}.npz"
    jnp.savez(
        save_path,
        H=H,
        K_truth=K,
        Ks=K_flat,
        alpha=alpha,
        beta=beta,
        Ls=Ls,
    )
    print(f"\nResults saved to {save_path}")
