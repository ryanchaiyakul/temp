from __future__ import annotations
from typing import NamedTuple
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
import jaxtyping

import hodel
import hodel.dismech as dismech


class TrainConfig(NamedTuple):
    method: hodel.Method
    nepochs: int
    # Optimizer
    lr: float
    # Solver
    nsteps: int = 5
    # Model
    prewarm: bool = False
    # Noise
    eta: float = 0.0


@register_dataclass
@dataclass(frozen=True)
class TripletAux:
    top: dismech.Connectivity  # for state.update()
    idx_f: jax.Array
    idx_b: jax.Array
    end_disp: jax.Array
    end_theta: jax.Array


def get_W_fn(
    mesh: dismech.Mesh, mass: jax.Array, g: jax.Array = jnp.array([0.0, 0.0, -9.81])
) -> Callable[[jax.Array, TripletAux], jax.Array]:
    def fn(_: jax.Array, aux: TripletAux) -> jax.Array:
        gravity = mass * jnp.concat(
            [
                jnp.tile(g, mesh.nodes.shape[0]),
                jnp.zeros(mesh.edges.shape[0]),
            ]
        )
        return gravity[aux.idx_f]

    return fn


def get_xb_fun(
    state0: dismech.StaticState,
) -> Callable[[jax.Array, TripletAux], jax.Array]:
    def fn(lambda_: jax.Array, aux: TripletAux) -> jax.Array:
        """Compress and contort."""
        return state0.q[aux.idx_b] + lambda_ * jnp.concat(
            [
                aux.end_disp,  # n0
                aux.end_disp,  # n1
                jnp.zeros(aux.idx_b.shape[0] - 8),
                jnp.array([aux.end_theta[0], 0.0]),
            ]
        )

    return fn


def get_indices(
    q: jax.Array,
    top: dismech.Connectivity,
    fixed_nodes: jax.Array | None = None,
    fixed_edges: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    nodes = (
        jnp.array([]) if fixed_nodes is None else dismech.map_node_to_dof(fixed_nodes)
    )
    edges = jnp.array([]) if fixed_edges is None else top.edge_dofs[0] + fixed_edges
    idx_b = jnp.union1d(nodes, edges)
    return idx_b, jnp.setdiff1d(jnp.arange(q.shape[0]), idx_b)


def update_state(
    xf: jax.Array, xb: jax.Array, aux: TripletAux, carry: dismech.StaticState
) -> dismech.StaticState:
    """Construct q from xf and xb and update state."""
    q = get_q(xf, xb, aux)
    carry_new = carry.update(q, aux.top)
    return carry_new


def get_energy_fn(
    triplets: dismech.Triplet,
) -> Callable[
    [jax.Array, jax.Array, jaxtyping.PyTree, TripletAux, dismech.StaticState], jax.Array
]:
    def get_batch_energy(
        xf: jax.Array,
        xb: jax.Array,
        Theta: jaxtyping.PyTree,
        aux: TripletAux,
        carry: dismech.StaticState,
    ) -> jax.Array:
        q = get_q(xf, xb, aux)
        state = carry.update(q, aux.top)
        return jnp.sum(jax.vmap(lambda t: t.get_energy(state, Theta))(triplets))

    return get_batch_energy


def get_q(xf: jax.Array, xb: jax.Array, aux: TripletAux) -> jax.Array:
    """Helper to construct q from xf and xb."""
    q = jnp.empty((aux.idx_f.shape[0] + aux.idx_b.shape[0]), xf.dtype)
    return q.at[aux.idx_f].set(xf).at[aux.idx_b].set(xb)


def animate_rod(sim: hodel.HODEL, lambdas: jax.Array, xf, aux):
    xb = jax.vmap(sim.get_xb, (0, None))(lambdas, aux)
    qs = jax.vmap(get_q, (0, 0, None))(xf, xb, aux)  # type: ignore
    return dismech.animate(lambdas, qs, aux.top)


def get_triplets(E: float = 1e7):
    geom = dismech.Geometry(
        rod_r0=0.001,
        shell_h=0,
    )

    mat = dismech.Material(
        density=1200,
        youngs_rod=E,
        youngs_shell=0,
        poisson_rod=0.5,
        poisson_shell=0,
    )

    mesh = dismech.Mesh.from_txt("data/rod.txt")

    top, state0, mass, K, triplets = dismech.from_legacy_custom(
        mesh, geom, mat, dismech.ParametrizedDERTriplet
    )
    K = jnp.array([K[0], K[2], K[3], K[4]])  # EA1 = EA2
    assert type(triplets) is dismech.ParametrizedDERTriplet

    idx_b, idx_f = get_indices(
        state0.q, top, jnp.array([0, 1, 19, 20]), jnp.array([0, 19])
    )
    get_W = get_W_fn(mesh, mass)
    get_xb = get_xb_fun(state0)
    sim = hodel.HODEL(
        get_energy_fn(triplets),
        get_W_fn=get_W,
        get_xb_fn=get_xb,
        carry_fn=update_state,
    )

    def get_aux(end_disp: jax.Array, end_theta: jax.Array) -> TripletAux:
        return TripletAux(top, idx_f, idx_b, end_disp, end_theta)

    return sim, state0, state0.q[idx_f], K, get_aux
