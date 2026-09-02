import numpy as np

from spde_surrogates.evaluation.metrics import (
    energy_distance_squared,
    timewise_energy_distance_squared,
    nearest_neighbor_relative_error,
)


def evaluate_surrogate(
    surrogate,
    real_trajectories,
    mus,
    group_ids,
    times,
    n_generated=100,
    temporal_coupling="shared_noise",
    verbose=True,
):
    """
    Evaluate a generative surrogate on several conditioning
    parameters mu.

    For every distinct group/mu:

        1. collect the corresponding real FOM trajectories;
        2. generate new surrogate trajectories;
        3. compute trajectory energy distance;
        4. compute time-wise energy distance;
        5. compute nearest-neighbor diagnostic.

    Parameters
    ----------
    surrogate
        A generative surrogate exposing

            surrogate.sample(
                mu,
                times,
                n_samples,
                temporal_coupling,
            )

    real_trajectories
        Test FOM trajectories.

        Shape:
            (N_real_total, Nt, Nh)

    mus
        Conditioning parameter associated with every
        real trajectory.

        Shape:
            (N_real_total, n_parameters)

    group_ids
        Identifier of the conditioning parameter mu
        associated with every real trajectory.

        Shape:
            (N_real_total,)

    times
        Physical time grid.

        Shape:
            (Nt,)

    n_generated
        Number of generated trajectories for each mu.

        This does NOT need to equal the number of
        available real trajectories.

    temporal_coupling
        "shared_noise" or "independent".

    verbose
        Print progress.

    Returns
    -------
    dict
        Contains results for every mu and aggregate
        statistics.
    """

    real_trajectories = np.asarray(
        real_trajectories
    )

    mus = np.asarray(
        mus
    )

    group_ids = np.asarray(
        group_ids
    )

    times = np.asarray(
        times
    )

    # --------------------------------------------------------
    # BASIC CHECKS
    # --------------------------------------------------------

    if real_trajectories.ndim != 3:
        raise ValueError(
            "real_trajectories must have shape "
            "(N, Nt, Nh)."
        )

    if mus.shape[0] != real_trajectories.shape[0]:
        raise ValueError(
            "mus and real_trajectories must contain "
            "the same number of trajectories."
        )

    if len(group_ids) != real_trajectories.shape[0]:
        raise ValueError(
            "group_ids and real_trajectories must contain "
            "the same number of trajectories."
        )

    if len(times) != real_trajectories.shape[1]:
        raise ValueError(
            "times and real_trajectories have incompatible "
            "time dimensions."
        )

    if n_generated <= 0:
        raise ValueError(
            "n_generated must be positive."
        )

    # --------------------------------------------------------
    # EVALUATE EVERY DISTINCT MU
    # --------------------------------------------------------

    unique_groups = np.unique(
        group_ids
    )

    per_group_results = []

    for k, group in enumerate(
        unique_groups
    ):

        mask = (
            group_ids == group
        )

        real_mu = real_trajectories[
            mask
        ]

        mus_mu = mus[
            mask
        ]

        # All trajectories belonging to the same group
        # must have the same conditioning parameter.
        mu = mus_mu[0]

        if not np.allclose(
            mus_mu,
            mu[None, :],
        ):
            raise ValueError(
                f"Group {group} contains multiple "
                "different mu values."
            )

        # ----------------------------------------------------
        # GENERATE SURROGATE TRAJECTORIES
        # ----------------------------------------------------

        generated_mu = surrogate.sample(
            mu=mu,
            times=times,
            n_samples=n_generated,
            temporal_coupling=temporal_coupling,
        )

        # ----------------------------------------------------
        # MAIN DISTRIBUTIONAL METRIC
        # ----------------------------------------------------

        energy = energy_distance_squared(
            real_mu,
            generated_mu,
        )

        # ----------------------------------------------------
        # TIME-WISE DISTRIBUTIONAL METRIC
        # ----------------------------------------------------

        timewise_energy = (
            timewise_energy_distance_squared(
                real_mu,
                generated_mu,
            )
        )

        # ----------------------------------------------------
        # SECONDARY NEAREST-NEIGHBOR DIAGNOSTIC
        # ----------------------------------------------------

        nn_result = (
            nearest_neighbor_relative_error(
                real_mu,
                generated_mu,
            )
        )

        result = {
            "group_id":
                int(group),

            "mu":
                mu.copy(),

            "n_real":
                int(
                    real_mu.shape[0]
                ),

            "n_generated":
                int(n_generated),

            "energy_distance_squared":
                float(energy),

            "timewise_energy_distance_squared":
                timewise_energy,

            "nearest_neighbor":
                nn_result,
        }

        per_group_results.append(
            result
        )

        if verbose:

            print(
                f"group "
                f"{k + 1:3d}/"
                f"{len(unique_groups)} "
                f"(id={group}) - "
                f"Nreal={real_mu.shape[0]} - "
                f"Ngen={n_generated} - "
                f"D2={energy:.6e}"
            )

    # --------------------------------------------------------
    # AGGREGATE OVER MU
    # --------------------------------------------------------

    energies = np.array(
        [
            result[
                "energy_distance_squared"
            ]
            for result
            in per_group_results
        ]
    )

    timewise_matrix = np.stack(
        [
            result[
                "timewise_energy_distance_squared"
            ]
            for result
            in per_group_results
        ]
    )

    nn_means = np.array(
        [
            result[
                "nearest_neighbor"
            ]["mean"]
            for result
            in per_group_results
        ]
    )

    summary = {
        "n_groups":
            int(
                len(unique_groups)
            ),

        "energy_mean":
            float(
                energies.mean()
            ),

        "energy_median":
            float(
                np.median(
                    energies
                )
            ),

        "energy_min":
            float(
                energies.min()
            ),

        "energy_max":
            float(
                energies.max()
            ),

        "energy_std":
            float(
                energies.std()
            ),

        "nearest_neighbor_mean":
            float(
                nn_means.mean()
            ),

        "timewise_mean":
            timewise_matrix.mean(
                axis=0
            ),

        "timewise_median":
            np.median(
                timewise_matrix,
                axis=0,
            ),

        "timewise_min":
            timewise_matrix.min(
                axis=0
            ),

        "timewise_max":
            timewise_matrix.max(
                axis=0
            ),
    }

    return {
        "per_group":
            per_group_results,

        "summary":
            summary,

        "times":
            times.copy(),
    }