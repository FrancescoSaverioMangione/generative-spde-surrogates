import os
import pickle

from pathlib import Path

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


def build_fom_dataset_resumable(
    problem,
    n_mu,
    n_realizations,
    nimp,
    dt,
    steps,
    checkpoint_dir,
    seed=None,
    max_new_trajectories=None,
):
    """
    Build a FOM dataset with trajectory-level checkpointing.

    The generation can be interrupted and resumed by calling
    this function again with exactly the same configuration
    and checkpoint directory.

    Each completed trajectory is saved separately. The NumPy
    random state is also stored after every completed
    trajectory, so stochastic sampling continues from the
    correct point after a restart.

    Parameters
    ----------
    problem
        Module implementing the physical problem.

    n_mu : int
        Number of different conditioning parameters.

    n_realizations : int
        Number of stochastic realizations for each mu.

    nimp : int
        Number of impurities.

    dt : float
        Physical time step.

    steps : int
        Number of time steps.

    checkpoint_dir : str or Path
        Persistent directory used for resumable generation.

    seed : int or None
        NumPy random seed.

    max_new_trajectories : int or None
        Optional maximum number of new trajectories generated
        during the current call. Mainly useful for testing the
        resume mechanism.

    Returns
    -------
    tuple or None
        Once the dataset is complete, returns

        (
            mu_bank,
            times,
            conds,
            trajectories,
            group_ids,
        )

        If the call stops early because of
        max_new_trajectories, returns None.
    """

    checkpoint_dir = Path(
        checkpoint_dir
    )

    trajectories_dir = (
        checkpoint_dir
        / "trajectories"
    )

    metadata_path = (
        checkpoint_dir
        / "metadata.pkl"
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trajectories_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_trajectories = (
        int(n_mu)
        * int(n_realizations)
    )

    # ========================================================
    # HELPERS
    # ========================================================

    def save_metadata_atomic(
        metadata,
    ):
        """
        Atomically save checkpoint metadata.
        """

        temporary_path = (
            checkpoint_dir
            / "metadata.tmp"
        )

        with temporary_path.open(
            "wb"
        ) as file:
            pickle.dump(
                metadata,
                file,
            )

        os.replace(
            temporary_path,
            metadata_path,
        )

    def save_trajectory_atomic(
        trajectory,
        path,
    ):
        """
        Atomically save one trajectory.
        """

        temporary_path = Path(
            str(path) + ".tmp"
        )

        with temporary_path.open(
            "wb"
        ) as file:
            np.save(
                file,
                trajectory,
            )

        os.replace(
            temporary_path,
            path,
        )

    # ========================================================
    # NEW DATASET OR RESUME
    # ========================================================

    if metadata_path.is_file():

        with metadata_path.open(
            "rb"
        ) as file:
            metadata = pickle.load(
                file
            )

        # ----------------------------------------------------
        # Protect against resuming with a different config.
        # ----------------------------------------------------

        expected = {
            "n_mu":
                int(n_mu),

            "n_realizations":
                int(n_realizations),

            "nimp":
                int(nimp),

            "dt":
                float(dt),

            "steps":
                int(steps),

            "seed":
                seed,
        }

        for key, expected_value in (
            expected.items()
        ):

            saved_value = metadata[
                key
            ]

            if saved_value != expected_value:
                raise ValueError(
                    "Checkpoint configuration "
                    f"mismatch for '{key}': "
                    f"saved={saved_value}, "
                    f"requested={expected_value}."
                )

        mu_bank = metadata[
            "mu_bank"
        ]

        completed = int(
            metadata["completed"]
        )

        # Verify that all trajectories declared completed
        # actually exist.
        for index in range(
            completed
        ):

            trajectory_path = (
                trajectories_dir
                / f"trajectory_{index:05d}.npy"
            )

            if not trajectory_path.is_file():
                raise RuntimeError(
                    "Checkpoint metadata reports "
                    f"trajectory {index} as completed, "
                    "but its file is missing."
                )

        # Restore stochastic state exactly where the previous
        # call stopped.
        np.random.set_state(
            metadata["rng_state"]
        )

        print(
            "Resuming dataset generation."
        )

        print(
            f"Already completed: "
            f"{completed}/"
            f"{total_trajectories}"
        )

    else:

        # ----------------------------------------------------
        # Initialize a new experiment.
        # ----------------------------------------------------

        if seed is not None:
            np.random.seed(
                seed
            )

        mu_bank = np.stack(
            [
                problem.sample_mu()
                for _ in range(n_mu)
            ]
        )

        completed = 0

        metadata = {
            "version":
                1,

            "n_mu":
                int(n_mu),

            "n_realizations":
                int(n_realizations),

            "nimp":
                int(nimp),

            "dt":
                float(dt),

            "steps":
                int(steps),

            "seed":
                seed,

            "mu_bank":
                mu_bank,

            "completed":
                completed,

            "rng_state":
                np.random.get_state(),
        }

        save_metadata_atomic(
            metadata
        )

        print(
            "Initialized new resumable dataset."
        )

        print(
            f"Total trajectories: "
            f"{total_trajectories}"
        )

    # ========================================================
    # GENERATION
    # ========================================================

    generated_this_call = 0

    while completed < total_trajectories:

        if (
            max_new_trajectories
            is not None
            and generated_this_call
            >= int(
                max_new_trajectories
            )
        ):

            print(
                "Stopping current call after "
                f"{generated_this_call} new "
                "trajectories."
            )

            print(
                f"Progress saved: "
                f"{completed}/"
                f"{total_trajectories}"
            )

            return None

        mu_index = (
            completed
            // n_realizations
        )

        realization_index = (
            completed
            % n_realizations
        )

        mu = mu_bank[
            mu_index
        ]

        # Generate stochastic realization.
        permeability = (
            problem.sample_permeability(
                nimp
            )
        )

        # Full-order solve.
        U = problem.FOMsolver(
            mu,
            permeability,
            dt=dt,
            steps=steps,
        )

        trajectory_path = (
            trajectories_dir
            / (
                f"trajectory_"
                f"{completed:05d}.npy"
            )
        )

        # Save trajectory BEFORE updating metadata.
        #
        # If interruption occurs between these two operations,
        # the trajectory is simply regenerated and overwritten
        # at the next run.
        save_trajectory_atomic(
            U,
            trajectory_path,
        )

        completed += 1

        generated_this_call += 1

        metadata[
            "completed"
        ] = completed

        metadata[
            "rng_state"
        ] = np.random.get_state()

        save_metadata_atomic(
            metadata
        )

        print(
            f"mu "
            f"{mu_index + 1:3d}/{n_mu}, "
            f"realizzazione "
            f"{realization_index + 1:2d}/"
            f"{n_realizations}, "
            f"totale "
            f"{completed:4d}/"
            f"{total_trajectories}, "
            f"U.shape={U.shape}"
        )

    # ========================================================
    # RECONSTRUCT FINAL DATASET
    # ========================================================

    print(
        "All trajectories generated."
    )

    print(
        "Reconstructing final arrays..."
    )

    trajectories = np.stack(
        [
            np.load(
                trajectories_dir
                / (
                    f"trajectory_"
                    f"{index:05d}.npy"
                )
            )
            for index in range(
                total_trajectories
            )
        ]
    )

    conds = np.repeat(
        mu_bank,
        n_realizations,
        axis=0,
    )

    group_ids = np.repeat(
        np.arange(
            n_mu
        ),
        n_realizations,
    )

    times = (
        np.arange(
            steps + 1
        )
        * dt
    )

    print(
        "Dataset reconstruction complete."
    )

    return (
        mu_bank,
        times,
        conds,
        trajectories,
        group_ids,
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