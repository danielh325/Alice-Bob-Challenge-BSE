"""Run the two-mode inflation (Tasks 1.1 to 1.3)."""

from collections.abc import Sequence

import dynamiqs as dq
from dynamiqs import QArray, TimeQArray
from dynamiqs.result import MESolveResult
from jax.typing import ArrayLike

from catqubit.models import two_mode_model
from catqubit.params import CatParams

# dynamiqs' default (1e-6) leaks spurious, even negative, population into the top
# Fock levels at N_a >= 28 and biases the convergence study; 1e-9 costs no extra time.
DEFAULT_SOLVER_TOL = 1e-9


def inflate(
    na: int,
    nb: int,
    params: CatParams,
    tsave: ArrayLike,
    memory0: QArray | None = None,
    extra_exp_ops: Sequence[QArray] = (),
    save_states: bool = True,
    solver_tol: float = DEFAULT_SOLVER_TOL,
    extra_H: QArray | TimeQArray | None = None,
) -> MESolveResult:
    """Evolve memory0 (default vacuum) with the buffer in vacuum under the full model.

    result.expects holds [<a^dag a>, <b^dag b>, *extra_exp_ops] over tsave. With
    save_states, result.states keeps the joint density matrix at every time for
    Wigner snapshots and partial traces; without it, only the final state is kept.
    solver_tol sets both rtol and atol of the Tsit5 integrator. extra_H (constant or
    time-dependent, e.g. a preparation pulse) is added to H_2ph.
    """
    memory0 = dq.fock(na, 0) if memory0 is None else memory0
    if memory0.shape[-2] != na:
        raise ValueError(f"memory0 dimension {memory0.shape[-2]} does not match na={na}")
    model = two_mode_model(na, nb, params)
    a, b = dq.destroy(na, nb)
    psi0 = dq.tensor(memory0, dq.fock(nb, 0))
    H = model.H if extra_H is None else model.H + extra_H
    return dq.mesolve(
        H,
        model.jump_ops,
        psi0,
        tsave,
        exp_ops=[a.dag() @ a, b.dag() @ b, *extra_exp_ops],
        method=dq.method.Tsit5(rtol=solver_tol, atol=solver_tol),
        save_states=save_states,
        progress_meter=False,
    )
