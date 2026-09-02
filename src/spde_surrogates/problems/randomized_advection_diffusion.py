# Physical problem:
# Time-dependent randomized advection-diffusion problem
# driven by a Brinkman/Stokes velocity field.

from dlroms import *
import numpy as np

from fenics import dx


# ============================================================
# DOMAIN AND BOUNDARY CONDITIONS
# ============================================================

domain = (
    fe.rectangle((0, 0), (3, 2))
    - fe.circle((0.95, 0.70), 0.25)
    - fe.circle((1.50, 1.35), 0.25)
    - fe.circle((2.05, 0.70), 0.25)
)

mesh = fe.mesh(domain, stepsize=0.05)


left = lambda x: x[0] < 1e-6
right = lambda x: 3 - x[0] < 1e-12


def circle(x0, r):
    return lambda x: (
        (x[0] - x0[0]) ** 2 + (x[1] - x0[1]) ** 2
    ) ** 0.5 < r + 1e-3


def inflows(b0, b1, b2, b3):

    return [
        [
            left,
            lambda x: [
                b0 * np.exp(-25 * (x[1] - 1) ** 2),
                0.0,
            ],
        ],
        [
            circle((0.95, 0.70), 0.25),
            lambda x: [
                b1 * (-x[1] + 0.70),
                b1 * (x[0] - 0.95),
            ],
        ],
        [
            circle((1.50, 1.35), 0.25),
            lambda x: [
                b2 * (-x[1] + 1.35),
                b2 * (x[0] - 1.50),
            ],
        ],
        [
            circle((2.05, 0.70), 0.25),
            lambda x: [
                b3 * (-x[1] + 0.70),
                b3 * (x[0] - 2.05),
            ],
        ],
    ]


outflows = [right]


# ============================================================
# BRINKMAN / STOKES SOLVER
# ============================================================

def brinkman_solver(
    mesh,
    inflows,
    outflows,
    permeability,
    source=[0.0, 0.0],
):

    from fenics import (
        FiniteElement,
        NodalEnrichedElement,
        FunctionSpace,
        VectorElement,
        TrialFunctions,
        TestFunctions,
        inner,
        grad,
        dx,
        div,
        assemble,
        DirichletBC,
        Constant,
    )

    from scipy.sparse.linalg import spsolve
    from scipy.sparse import csr_matrix

    pP1 = FiniteElement(
        "CG",
        mesh.ufl_cell(),
        1,
    )

    vP1B = VectorElement(
        NodalEnrichedElement(
            FiniteElement(
                "CG",
                mesh.ufl_cell(),
                1,
            ),
            FiniteElement(
                "Bubble",
                mesh.ufl_cell(),
                mesh.topology().dim() + 1,
            ),
        )
    )

    pspace = pP1
    vspace = vP1B

    W = FunctionSpace(
        mesh,
        vspace * pspace,
    )

    W0 = FunctionSpace(
        mesh,
        vspace,
    )

    b, p = TrialFunctions(W)
    v, q = TestFunctions(W)

    space = fe.space(
        mesh,
        "CG",
        1,
        vector_valued=True,
        bubble=True,
    )

    dspace = fe.space(
        mesh,
        "DG",
        0,
    )

    f = fe.interpolate(
        source,
        space,
    )

    kappainv = fe.asfunction(
        1.0 / permeability,
        dspace,
    )

    a = (
        inner(grad(b), grad(v)) * dx
        - div(v) * p * dx
        - q * div(b) * dx
        + kappainv * inner(b, v) * dx
    )

    L = inner(f, v) * dx

    def outflow(x):
        result = False

        for out in outflows:
            result = result or out(x)

        return result

    def inflow(x):
        result = False

        for infl in inflows:
            result = result or infl[0](x)

        return result

    def walls(x):
        return not (
            outflow(x)
            or inflow(x)
        )

    noslip = DirichletBC(
        W.sub(0),
        Constant((0.0, 0.0)),
        lambda x, on: on and walls(x),
    )

    def make_bc(i):

        return DirichletBC(
            W.sub(0),
            fe.interpolate(
                inflows[i][1],
                W0,
            ),
            lambda x, on: (
                on
                and inflows[i][0](x)
            ),
        )

    ins = [
        make_bc(i)
        for i in range(len(inflows))
    ]

    A = assemble(a)
    F = assemble(L)

    for bc in [noslip, *ins]:

        bc.apply(A)
        bc.apply(F)

    A = csr_matrix(
        A.array()
    )

    F = F[:]

    bp = spsolve(
        A,
        F,
    )

    bp_f = fe.asfunction(
        bp,
        W,
    )

    from fenics import dof_to_vertex_map

    Vh_local = fe.space(
        mesh,
        "CG",
        1,
    )

    nvertices = (
        mesh
        .coordinates()
        .shape[0]
    )

    b = (
        bp_f
        .compute_vertex_values(mesh)[:2 * nvertices]
        .reshape(2, -1)
        .T
    )

    indexes = dof_to_vertex_map(
        Vh_local
    )

    b = b[indexes].reshape(-1)

    return b


# ============================================================
# FUNCTION SPACES
# ============================================================

Vh = fe.space(
    mesh,
    "CG",
    1,
)

Vb = fe.space(
    mesh,
    "CG",
    1,
    vector_valued=True,
)

Dh = fe.space(
    mesh,
    "DG",
    0,
)


M = fe.assemble(
    lambda u, v: u * v * dx,
    Vh,
)


# ============================================================
# TIME-DEPENDENT FOM
# ============================================================

def steadyFOMsolver(
    mu,
    permeability,
    dt,
    u0,
    b,
):

    from fenics import (
        grad,
        inner,
        Constant,
    )

    from scipy.sparse.linalg import spsolve

    def a(u, v):

        return (
            Constant(dt * 0.005)
            * inner(
                grad(u),
                grad(v),
            )
            * dx
            + Constant(dt)
            * inner(
                b,
                grad(u),
            )
            * v
            * dx
            + inner(u, v)
            * dx
        )

    A = fe.assemble(
        a,
        Vh,
    )

    F = M @ u0

    bc = fe.DirichletBC(
        left,
        lambda x: np.exp(
            -25
            * (x[1] - 1) ** 2
        ),
    )

    A = fe.applyBCs(
        A,
        Vh,
        bc,
    )

    F = fe.applyBCs(
        F,
        Vh,
        bc,
    )

    u = spsolve(
        A,
        F,
    )

    return u


def FOMsolver(
    mu,
    permeability,
    dt=1e-3,
    u0=None,
    steps=50,
):

    if u0 is None:
        u0 = np.zeros(
            Vh.dim()
        )

    u = [u0]

    velocity = brinkman_solver(
        mesh,
        inflows(
            1.0,
            *mu,
        ),
        outflows,
        permeability,
    )

    b = fe.asfunction(
        velocity,
        Vb,
    )

    for _ in range(steps):

        u.append(
            steadyFOMsolver(
                mu,
                permeability,
                dt,
                u[-1],
                b,
            )
        )

    return np.stack(u)


# ============================================================
# STOCHASTIC PERMEABILITY FIELD
# ============================================================

rho = 0.1

boundary = (
    mesh.coordinates()[
        fe.boundary(mesh)
    ]
)


def sample_impurity():

    found = False

    while not found:

        candidate = np.random.uniform(
            low=(0, 0),
            high=(3, 2),
            size=(2,),
        )

        dist = np.linalg.norm(
            boundary
            - candidate.reshape(1, 2),
            axis=-1,
        ).min()

        found = dist > rho

    return candidate


def sample_permeability(nimp):

    impurities = np.stack(
        [
            sample_impurity()
            for _ in range(nimp)
        ]
    )

    xdofs = fe.dofs(Dh)

    distances = np.linalg.norm(
        xdofs.reshape(-1, 1, 2)
        - impurities.reshape(
            1,
            -1,
            2,
        ),
        axis=-1,
    ).min(axis=1)

    permeability = (
        1000
        * np.ones(
            Dh.dim()
        )
    )

    permeability[
        distances < rho
    ] = 0.001

    return permeability


# ============================================================
# CONDITIONING PARAMETER
# ============================================================

def sample_mu(
    low=-6.0,
    high=6.0,
):

    return np.random.uniform(
        low=low,
        high=high,
        size=(3,),
    )

# ============================================================
# QUALITATIVE VISUALIZATION
# ============================================================

def animate_trajectory_comparison(
    trajectories,
    figsize=(15, 4),
    cmap="turbo",
    colorbar=True,
):
    """
    Animate several trajectories of the randomized
    advection-diffusion problem side by side.

    Parameters
    ----------
    trajectories
        Iterable containing trajectories with shape:
            (Nt, Nh)

        Example:
            (
                real_trajectory,
                nf_trajectory,
                fm_trajectory,
            )

    Returns
    -------
    Animation produced by dlroms.fe.animate.
    """

    trajectories = tuple(
        trajectories
    )

    return fe.animate(
        trajectories,
        Vh,
        figsize=figsize,
        cmap=cmap,
        colorbar=colorbar,
    )