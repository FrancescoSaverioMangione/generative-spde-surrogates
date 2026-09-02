import numpy as np
import matplotlib.pyplot as plt


def select_high_variance_probe(
    real_trajectories,
):
    """
    Select the spatial DOF with the largest variance
    across all real trajectories and physical times.

    Parameters
    ----------
    real_trajectories
        Shape:
            (N_real, Nt, Nh)

    Returns
    -------
    int
        Selected spatial DOF.
    """

    real_trajectories = np.asarray(
        real_trajectories
    )

    if real_trajectories.ndim != 3:
        raise ValueError(
            "real_trajectories must have shape "
            "(N_real, Nt, Nh)."
        )

    n_dofs = real_trajectories.shape[-1]

    flattened = real_trajectories.reshape(
        -1,
        n_dofs,
    )

    variance = np.var(
        flattened,
        axis=0,
    )

    return int(
        np.argmax(variance)
    )


def plot_temporal_probe_comparison(
    times,
    real_trajectories,
    generated_sets,
    probe=None,
    max_real=20,
    title=None,
):
    """
    Compare real and generated temporal evolutions
    at one spatial degree of freedom.

    Parameters
    ----------
    times
        Shape:
            (Nt,)

    real_trajectories
        Shape:
            (N_real, Nt, Nh)

    generated_sets
        Dictionary such as:

            {
                "NF": nf_samples,
                "FM": fm_samples,
            }

        Each array must have shape:
            (N_generated, Nt, Nh)

    probe
        Spatial DOF to inspect.
        If None, the highest-variance real DOF is selected.

    max_real
        Maximum number of real trajectories shown.

    title
        Optional figure title.

    Returns
    -------
    fig, ax, probe
    """

    times = np.asarray(times)
    real_trajectories = np.asarray(
        real_trajectories
    )

    if real_trajectories.ndim != 3:
        raise ValueError(
            "real_trajectories must have shape "
            "(N_real, Nt, Nh)."
        )

    if probe is None:
        probe = select_high_variance_probe(
            real_trajectories
        )

    fig, ax = plt.subplots(
        figsize=(9, 4)
    )

    # Plot real FOM realizations.
    n_show = min(
        max_real,
        real_trajectories.shape[0],
    )

    for i in range(n_show):

        ax.plot(
            times,
            real_trajectories[
                i,
                :,
                probe,
            ],
            linewidth=1.0,
            alpha=0.25,
        )

    # Plot the mean generated trajectory
    # for every generative model.
    for name, samples in (
        generated_sets.items()
    ):

        samples = np.asarray(
            samples
        )

        if samples.ndim != 3:
            raise ValueError(
                f"{name} samples must have shape "
                "(N_generated, Nt, Nh)."
            )

        generated_mean = samples[
            :,
            :,
            probe,
        ].mean(
            axis=0
        )

        ax.plot(
            times,
            generated_mean,
            linewidth=2.5,
            label=f"{name} generated mean",
        )

    ax.set_xlabel(
        "physical time"
    )

    ax.set_ylabel(
        f"u at DOF #{probe}"
    )

    if title is None:
        title = (
            "Temporal comparison at "
            f"spatial DOF #{probe}"
        )

    ax.set_title(title)

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    return (
        fig,
        ax,
        probe,
    )


def plot_timewise_energy_comparison(
    times,
    evaluations,
):
    """
    Compare the mean time-wise energy-distance curves
    of several surrogate models.

    Parameters
    ----------
    times
        Physical time grid.

    evaluations
        Dictionary such as:

            {
                "NF": nf_evaluation,
                "FM": fm_evaluation,
            }

    Returns
    -------
    fig, ax
    """

    times = np.asarray(
        times
    )

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    for name, evaluation in (
        evaluations.items()
    ):

        values = evaluation[
            "summary"
        ]["timewise_mean"]

        ax.plot(
            times,
            values,
            linewidth=2,
            label=name,
        )

    ax.set_xlabel(
        "physical time"
    )

    ax.set_ylabel(
        "mean time-wise energy distance squared"
    )

    ax.set_title(
        "Distributional error over physical time"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    return fig, ax