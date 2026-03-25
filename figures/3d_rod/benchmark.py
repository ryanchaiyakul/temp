from __future__ import annotations
from typing import NamedTuple
import argparse

import jax
import jax.numpy as jnp
import optax
import tqdm

import hodel
from helper import get_triplets

jax.config.update("jax_enable_x64", True)


class TrainConfig(NamedTuple):
    method: hodel.Method
    lr: float
    nepochs: int


pinn_config = TrainConfig(hodel.Method.PINN, 1e-5, 200)
pinn_03_config = TrainConfig(hodel.Method.PINN, 1e-6, 200)
hodel_config = TrainConfig(hodel.Method.HODEL, 1e-7, 200)
deq_config = TrainConfig(hodel.Method.DEQ, 1e-6, 700)
node_config = TrainConfig(hodel.Method.ODE, 1e-6, 24)
config_map = {
    "pinn": pinn_config,
    "hodel": hodel_config,
    "deq": deq_config,
    "node": node_config,
}


def clip_strictly_positive(eps=1e-9):
    return optax.stateless(lambda p, _: jnp.maximum(p, eps))  # type: ignore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 1D benchmark.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["hodel", "pinn", "deq", "node"],
        help="The learning framework to use.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of random K seeds to run (default: 30).",
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.001,
        help="Eta noise scaling factor (defaults: 0.001).",
    )
    args = parser.parse_args()
    config = config_map[args.method]

    if args.eta >= 0.03 and config == pinn_config:
        config = pinn_03_config

    sim, state, xf0, K, get_aux = get_triplets(1e6)

    # Does not converge smaller
    if args.method == "deq":
        lambdas = jnp.linspace(0, 0.4, 20)
    else:
        lambdas = jnp.linspace(0, 1.0, 50)

    aux = get_aux(jnp.array([0.005, 0.005, -0.005]), jnp.array([0.5]))
    train_aux = get_aux(jnp.array([0.01, 0.0, 0.0]), jnp.array([1.0]))
    xf_stars = sim.solve(lambdas, xf0, K, aux, state)
    xf_train = sim.solve(lambdas, xf0, K, train_aux, state)
    key = jax.random.PRNGKey(42)

    def get_perturbed_K(key, K, sigma_values):
        subkeys = jax.random.split(key, len(sigma_values))

        def perturb(subkey, sigma):
            z = jax.random.normal(subkey, K.shape) * sigma
            return K * jnp.exp(z - (sigma**2) / 2.0)

        return jax.vmap(perturb)(subkeys, sigma_values)

    eta_range = jnp.repeat(0.15, args.seeds)
    K_perturbed = get_perturbed_K(key, K, eta_range)

    xf_train_noise = xf_train * (1 + args.eta * jax.random.normal(key, xf_train.shape))

    @jax.jit
    def run(K_: jax.Array, subkey: jax.Array):
        noise = 1 + args.eta * jax.random.normal(subkey, xf_train.shape)
        xf_train_noise = xf_train * noise

        def eval_fn(Theta):
            loss = sim.loss(lambdas, xf0, xf_stars, Theta, aux, state)
            jax.debug.print("{}: {}", Theta, loss)
            return loss

        bounded_optim = optax.chain(optax.adam(config.lr), clip_strictly_positive())

        K, _, L_test = sim.learn_and_eval(
            lambdas,
            xf0,
            xf_train_noise,
            K_,
            train_aux,
            state,
            optim=bounded_optim,
            nepochs=config.nepochs,
            method=config.method,
            eval_fn=eval_fn,
        )
        return K, L_test

    all_test_errors = []
    for i, K_ in enumerate(tqdm.tqdm(K_perturbed)):
        iter_key = jax.random.fold_in(key, i)
        _, L_test = run(K_, iter_key)
        all_test_errors.append(L_test)

    filename = f"data/benchmarks_{args.method}_eta={args.eta}.npz"
    jnp.savez(
        filename,
        test_errors=jnp.array(all_test_errors),
        method=args.method,
    )
    print(f"Successfully saved results to {filename}")
