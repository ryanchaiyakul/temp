from typing import NamedTuple
import argparse

import jax
import jax.numpy as jnp
import optax
import tqdm

import hodel
from helper import get_triplets

jax.config.update("jax_enable_x64", True)


bounded_optim = optax.chain(
    optax.adam(1e-3),
)


class TrainConfig(NamedTuple):
    method: hodel.Method
    nepochs: int


pinn_config = TrainConfig(hodel.Method.PINN, 1000)
hodel_config = TrainConfig(hodel.Method.HODEL, 50)
node_config = TrainConfig(hodel.Method.ODE, 60)
config_map = {
    "pinn": pinn_config,
    "hodel": hodel_config,
    "node": node_config,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run 3D benchmark.")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["hodel", "pinn", "node"],
        help="The learning framework to use.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=30,
        help="Number of random K seeds to run (default: 30).",
    )
    args = parser.parse_args()
    config = config_map[args.method]
    seeds = jax.vmap(lambda i: jax.random.PRNGKey(i))(jnp.arange(args.seeds))
    final_params = []

    def train(key: jax.Array):
        sim, state0, xf0, params, get_aux = get_triplets(key=key, icnn=False)
        data = jnp.load("data/ribbon_data.npz")
        train_xf_stars = data["train_xf_stars"]
        train_aux = get_aux(data["train_aux"][0])
        lambdas = jnp.linspace(0, 1.0, 11)
        final_params, _ = sim.learn_lbfgs(
            lambdas,
            xf0,
            train_xf_stars[0],
            params,
            train_aux,
            state0,
            method=config.method,
            #optim=bounded_optim,
            nepochs=config.nepochs,
            config=hodel.SolverConfig(isteps=200),
        )
        return final_params

    for k in tqdm.tqdm(seeds):
        final_params.append(train(k))

    sim, state0, xf0, _, get_aux = get_triplets(icnn=False)  # ignore initial params
    lambdas = jnp.linspace(0.0, 1.0, 11)

    @jax.jit
    def fn(params_, batch_xf_stars, batch_aux):
        return jax.vmap(
            lambda xf_stars_, aux_: sim.loss(
                lambdas,
                xf0,
                xf_stars_,
                params_,
                aux_,
                state0,
                config=hodel.SolverConfig(isteps=200),
            )
        )(batch_xf_stars, batch_aux)

    data = jnp.load("data/ribbon_data.npz")
    test_xf_stars = data["test_xf_stars"]
    test_aux = jax.vmap(get_aux)(data["test_aux"])
    batch_errs = []
    for p in final_params:
        batch_errs.append(fn(p, test_xf_stars, test_aux))
    batch_errs = jnp.asarray(batch_errs)
    jnp.savez(f"benchmarks_{args.method}.npz", final_params=final_params, batch_errs=batch_errs)
