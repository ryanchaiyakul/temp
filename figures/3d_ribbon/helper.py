from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
import jaxtyping
import flax.linen as nn

import hodel
import hodel.dismech as dismech


class HoDELNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        init = nn.initializers.variance_scaling(0.01, "fan_in", "truncated_normal")
        w1 = self.param("w1", init, (x.shape[-1], self.hidden_size))
        x = nn.softplus(jnp.dot(x, jnp.abs(w1)))
        w2 = self.param("w2", init, (self.hidden_size, self.hidden_size))
        x = nn.softplus(jnp.dot(x, jnp.abs(w2)))
        wf = self.param(
            "wf", nn.initializers.truncated_normal(1e-4), (self.hidden_size, 1)
        )
        return jnp.squeeze(jnp.dot(x, jnp.abs(wf)))


class NN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        x = nn.softplus(nn.Dense(self.hidden_size)(x))
        x = nn.softplus(nn.Dense(self.hidden_size)(x))
        return nn.Dense(1)(x)[0] ** 2  # For positive


@register_dataclass
@dataclass(frozen=True)
class TripletAux:
    top: dismech.Connectivity  # for state.update()
    idx_f: jax.Array
    idx_b: jax.Array
    xb_m: jax.Array


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
        return state0.q[aux.idx_b] + lambda_ * aux.xb_m

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
    triplets: dismech.DERTriplet,
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


def animate_rod(sim: hodel.HODEL, lambdas, xf, aux):
    xb = jax.vmap(sim.get_xb, (0, None))(lambdas, aux)
    qs = jax.vmap(get_q, (0, 0, None))(xf, xb, aux)  # type: ignore
    return dismech.animate(lambdas, qs, aux.top)


def get_triplets(
    E: float = 1e6,
    w: float = 0.03,
    t: float = 0.001,
    key: jax.Array = jax.random.PRNGKey(42),
    der: bool = False,
    icnn: bool = True,
    from_x: bool = False,
    mesh_path: str = "data/ribbon.txt",
    solve=jnp.linalg.solve,
):
    geom = dismech.Geometry(
        rod_r0=0.0,
        shell_h=0.0,
        axs=w * t,
        ixs1=(w * t**3) / 12,
        ixs2=(t * w**3) / 12,
        jxs=(1 / 3) * w * (t**3),
    )

    mat = dismech.Material(
        density=1000,
        youngs_rod=E,
        youngs_shell=0,
        poisson_rod=0.5,
        poisson_shell=0,
    )
    if icnn:
        model = HoDELNN(10)
    else:
        model = NN(10)

    @register_dataclass
    @dataclass(frozen=True)
    class NNTriplet(dismech.DERTriplet):
        """3 node spring where Theta is parameters for a NN."""

        def get_psi(self, del_strain: jax.Array, Theta: jaxtyping.PyTree) -> jax.Array:
            k1, k2, tau = del_strain[2], del_strain[3], del_strain[4]
            features = jnp.array(
                [
                    k1**2,
                    k2**2,
                    tau**2,
                    (k1 * tau) ** 2,
                    (k2 * tau) ** 2,
                ]
            )
            return model.apply(Theta, features)  # type: ignore

    @register_dataclass
    @dataclass(frozen=True)
    class XtoENNTriplet(dismech.DERTriplet):
        def get_energy(
            self, state: dismech.StaticState, Theta: jaxtyping.PyTree
        ) -> jax.Array:
            return model.apply(Theta, state.q)  # type: ignore

    mesh = dismech.Mesh.from_txt(mesh_path)

    if from_x:
        factory = XtoENNTriplet.init
        params = model.init(key, jnp.empty(123))  # hard coded but state.q
    else:
        factory = NNTriplet.init
        params = model.init(key, jnp.empty(5))

    if der:
        top, state0, mass, triplets, _ = dismech.from_legacy(mesh, geom, mat)
    else:
        top, state0, mass, triplets, _ = dismech.from_legacy(
            mesh, geom, mat, triplet_factory=factory
        )
    assert isinstance(triplets, dismech.DERTriplet)
    idx_b, idx_f = get_indices(
        state0.q, top, jnp.array([0, 1, 29, 30]), jnp.array([0, 29])
    )

    get_W = get_W_fn(mesh, mass, jnp.array([0.0, 0.0, 1e-2]))
    get_xb = get_xb_fun(state0)
    sim = hodel.HODEL(
        get_energy_fn(triplets),
        get_W_fn=get_W,
        get_xb_fn=get_xb,
        carry_fn=update_state,
        linalg_solve=solve,
    )

    def get_aux(xb_m: jax.Array) -> TripletAux:
        return TripletAux(top, idx_f, idx_b, xb_m)

    return sim, state0, state0.q[idx_f], params, get_aux
