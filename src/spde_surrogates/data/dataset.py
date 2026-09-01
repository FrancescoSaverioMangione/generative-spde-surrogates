import os
import numpy as np


# ============================================================
# FOM DATASET GENERATION
# ============================================================

def build_fom_dataset(
    problem,
    n_mu,
    n_realizations,
    nimp,
    dt,
    steps,
    seed=None,
):
    """
    Build a dataset of FOM trajectories.

    Parameters
    ----------
    problem
        Module implementing the physical problem.
        It must provide:
        - sample_mu()
        - sample_permeability(nimp)
        - FOMsolver(mu, permeability, dt, steps)

    n_mu : int
        Number of different conditioning parameters mu.

    n_realizations : int
        Number of stochastic realizations for each mu.

    nimp : int
        Number of impurities used to generate the random permeability.

    dt : float
        Physical time step.

    steps : int
        Number of time steps.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    mu_bank
        Array with all distinct mu values.

    times
        Physical time grid.

    conds
        Mu associated with every generated trajectory.

    trajectories
        FOM trajectories.

    group_ids
        Identifier associating each trajectory with its mu.
    """

    if seed is not None:
        np.random.seed(seed)

    # Sample all distinct conditioning parameters.
    mu_bank = np.stack(
        [
            problem.sample_mu()
            for _ in range(n_mu)
        ]
    )

    conds = []
    trajectories = []
    group_ids = []

    for i, mu in enumerate(mu_bank):

        for r in range(n_realizations):

            # Generate one stochastic realization.
            permeability = problem.sample_permeability(
                nimp
            )

            # Solve the Full Order Model.
            U = problem.FOMsolver(
                mu,
                permeability,
                dt=dt,
                steps=steps,
            )

            conds.append(
                mu.copy()
            )

            trajectories.append(
                U
            )

            group_ids.append(
                i
            )

            print(
                f"mu {i + 1:3d}/{n_mu}, "
                f"realizzazione {r + 1:2d}/{n_realizations}, "
                f"U.shape={U.shape}"
            )

    # t_0 = 0, ..., t_steps = steps * dt
    times = (
        np.arange(steps + 1)
        * dt
    )

    return (
        mu_bank,
        times,
        np.stack(conds),
        np.stack(trajectories),
        np.array(group_ids),
    )


# ============================================================
# SAVE DATASET
# ============================================================

def save_fom_dataset(
    path,
    mu_bank,
    times,
    conds,
    trajectories,
    group_ids,
):
    """
    Save a FOM dataset in compressed NumPy format.
    """

    np.savez_compressed(
        path,
        mu_bank=mu_bank,
        times=times,
        conds=conds,
        trajectories=trajectories,
        group_ids=group_ids,
    )


# ============================================================
# LOAD DATASET
# ============================================================

def load_fom_dataset(path):
    """
    Load a previously generated FOM dataset.
    """

    data = np.load(path)

    return (
        data["mu_bank"],
        data["times"],
        data["conds"],
        data["trajectories"],
        data["group_ids"],
    )


# ============================================================
# LOAD OR GENERATE DATASET
# ============================================================

def get_fom_dataset(
    path,
    problem,
    n_mu,
    n_realizations,
    nimp,
    dt,
    steps,
    seed=None,
):
    """
    Load a dataset if it already exists.
    Otherwise generate it and save it.
    """

    if os.path.exists(path):

        print(
            f"Loading existing dataset from: {path}"
        )

        return load_fom_dataset(
            path
        )

    print(
        f"Dataset not found. Generating: {path}"
    )

    dataset = build_fom_dataset(
        problem=problem,
        n_mu=n_mu,
        n_realizations=n_realizations,
        nimp=nimp,
        dt=dt,
        steps=steps,
        seed=seed,
    )

    save_fom_dataset(
        path,
        *dataset,
    )

    return dataset