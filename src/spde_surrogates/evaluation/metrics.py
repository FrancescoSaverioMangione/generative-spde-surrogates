import numpy as np


def pairwise_distances(
    samples_a,
    samples_b,
):
    """
    Compute pairwise Euclidean/Frobenius distances.

    The first dimension identifies different samples.
    All remaining dimensions are flattened.

    Examples
    --------
    Trajectories:
        samples_a.shape = (N, Nt, Nh)
        samples_b.shape = (M, Nt, Nh)

    Snapshots at fixed time:
        samples_a.shape = (N, Nh)
        samples_b.shape = (M, Nh)

    Returns
    -------
    distances
        Shape (N, M).
    """

    samples_a = np.asarray(
        samples_a,
        dtype=np.float64,
    )

    samples_b = np.asarray(
        samples_b,
        dtype=np.float64,
    )

    if samples_a.ndim < 2:
        raise ValueError(
            "samples_a must contain a sample dimension "
            "and at least one feature dimension."
        )

    if samples_b.ndim < 2:
        raise ValueError(
            "samples_b must contain a sample dimension "
            "and at least one feature dimension."
        )

    A = samples_a.reshape(
        samples_a.shape[0],
        -1,
    )

    B = samples_b.reshape(
        samples_b.shape[0],
        -1,
    )

    if A.shape[1] != B.shape[1]:
        raise ValueError(
            "The two sample sets must have the same "
            "feature dimension."
        )

    norm_a = np.sum(
        A ** 2,
        axis=1,
        keepdims=True,
    )

    norm_b = np.sum(
        B ** 2,
        axis=1,
        keepdims=True,
    ).T

    squared_distances = (
        norm_a
        + norm_b
        - 2.0 * A @ B.T
    )

    # Numerical round-off can produce tiny negative values.
    squared_distances = np.maximum(
        squared_distances,
        0.0,
    )

    return np.sqrt(
        squared_distances
    )


def energy_distance_squared(
    real_samples,
    generated_samples,
):
    """
    Empirical trajectory/snapshot energy-distance quantity.

    Implements

        D^2 =
            2 E ||X - Y||
            - E ||X - X'||
            - E ||Y - Y'||

    using empirical Monte Carlo averages.

    real_samples and generated_samples are allowed to
    contain DIFFERENT numbers of samples.
    """

    real_samples = np.asarray(
        real_samples
    )

    generated_samples = np.asarray(
        generated_samples
    )

    if real_samples.shape[0] == 0:
        raise ValueError(
            "real_samples cannot be empty."
        )

    if generated_samples.shape[0] == 0:
        raise ValueError(
            "generated_samples cannot be empty."
        )

    d_real_generated = pairwise_distances(
        real_samples,
        generated_samples,
    )

    d_real_real = pairwise_distances(
        real_samples,
        real_samples,
    )

    d_generated_generated = pairwise_distances(
        generated_samples,
        generated_samples,
    )

    value = (
        2.0
        * d_real_generated.mean()
        - d_real_real.mean()
        - d_generated_generated.mean()
    )

    # The population quantity is non-negative.
    # Remove possible tiny negative values caused by
    # floating-point round-off.
    return float(
        max(value, 0.0)
    )


def timewise_energy_distance_squared(
    real_trajectories,
    generated_trajectories,
):
    """
    Compute the distributional error independently
    at every physical time.

    Parameters
    ----------
    real_trajectories
        Shape:
            (N_real, Nt, Nh)

    generated_trajectories
        Shape:
            (N_generated, Nt, Nh)

    Returns
    -------
    errors
        Shape:
            (Nt,)
    """

    real_trajectories = np.asarray(
        real_trajectories
    )

    generated_trajectories = np.asarray(
        generated_trajectories
    )

    if real_trajectories.ndim != 3:
        raise ValueError(
            "real_trajectories must have shape "
            "(N_real, Nt, Nh)."
        )

    if generated_trajectories.ndim != 3:
        raise ValueError(
            "generated_trajectories must have shape "
            "(N_generated, Nt, Nh)."
        )

    if (
        real_trajectories.shape[1:]
        != generated_trajectories.shape[1:]
    ):
        raise ValueError(
            "Real and generated trajectories must have "
            "the same time and spatial dimensions."
        )

    n_times = real_trajectories.shape[1]

    errors = np.zeros(
        n_times,
        dtype=np.float64,
    )

    for t in range(n_times):

        errors[t] = (
            energy_distance_squared(
                real_trajectories[:, t, :],
                generated_trajectories[:, t, :],
            )
        )

    return errors


def nearest_neighbor_relative_error(
    real_trajectories,
    generated_trajectories,
    eps=1e-12,
):
    """
    Secondary diagnostic.

    For every true trajectory, find the closest generated
    trajectory and compute the relative Frobenius error.

    This is NOT the main distributional metric.
    """

    real_trajectories = np.asarray(
        real_trajectories,
        dtype=np.float64,
    )

    generated_trajectories = np.asarray(
        generated_trajectories,
        dtype=np.float64,
    )

    distances = pairwise_distances(
        real_trajectories,
        generated_trajectories,
    )

    nearest_distances = distances.min(
        axis=1
    )

    real_flat = real_trajectories.reshape(
        real_trajectories.shape[0],
        -1,
    )

    real_norms = np.linalg.norm(
        real_flat,
        axis=1,
    )

    relative_errors = (
        nearest_distances
        / (
            real_norms
            + eps
        )
    )

    return {
        "mean":
            float(
                relative_errors.mean()
            ),

        "median":
            float(
                np.median(
                    relative_errors
                )
            ),

        "min":
            float(
                relative_errors.min()
            ),

        "max":
            float(
                relative_errors.max()
            ),

        "per_trajectory":
            relative_errors,
    }