import numpy as np

from spde_surrogates.data.splitting import split_by_group
from spde_surrogates.data.preprocessing import (
    ArrayNormalizer,
    build_static_samples,
)
from spde_surrogates.rom.pod import PODReducer
from spde_surrogates.models.realnvp import ConditionalRealNVP
from spde_surrogates.training.realnvp_trainer import train_realnvp


# ============================================================
# PREPARE DATA FOR A STATIC POD-BASED GENERATIVE MODEL
# ============================================================

def prepare_static_data(
    trajectories,
    conds,
    times,
    group_ids,
    split_config=None,
    pod_config=None,
):
    """
    Prepare train/validation/test data for a static
    POD-based conditional generative model.

    Pipeline:
        trajectories
            -> grouped train/val/test split
            -> POD fit on TRAIN only
            -> POD projection
            -> static samples conditioned on (mu, t)
            -> normalization fit on TRAIN only

    Parameters
    ----------
    trajectories
        Shape:
            (n_trajectories, n_times, n_dofs)

    conds
        Conditioning parameter mu for each trajectory.
        Shape:
            (n_trajectories, n_parameters)

    times
        Physical time grid.
        Shape:
            (n_times,)

    group_ids
        Group identifier associated with each trajectory.

    split_config
        Optional dictionary containing:
            train_fraction
            val_fraction
            test_fraction
            seed

    pod_config
        Optional dictionary passed to PODReducer.

    Returns
    -------
    dict
        Everything needed for model training and later
        reconstruction/sampling.
    """

    trajectories = np.asarray(trajectories)
    conds = np.asarray(conds)
    times = np.asarray(times)
    group_ids = np.asarray(group_ids)

    # --------------------------------------------------------
    # DEFAULT CONFIGURATIONS
    # --------------------------------------------------------

    if split_config is None:
        split_config = {
            "train_fraction": 0.70,
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "seed": 1,
        }

    if pod_config is None:
        pod_config = {
            "target_error": 1e-2,
            "max_modes": 100,
            "selection": "trajectory_error",
        }

    # --------------------------------------------------------
    # GROUPED SPLIT
    # --------------------------------------------------------

    split = split_by_group(
        group_ids=group_ids,
        **split_config,
    )

    idx_train = split["train_indices"]
    idx_val = split["val_indices"]
    idx_test = split["test_indices"]

    trajectories_train = trajectories[idx_train]
    trajectories_val = trajectories[idx_val]
    trajectories_test = trajectories[idx_test]

    mus_train = conds[idx_train]
    mus_val = conds[idx_val]
    mus_test = conds[idx_test]

    # --------------------------------------------------------
    # POD
    #
    # IMPORTANT:
    # fit ONLY on training trajectories.
    # --------------------------------------------------------

    pod = PODReducer(
        **pod_config
    )

    pod.fit(
        trajectories_train
    )

    coeff_train = pod.transform(
        trajectories_train
    )

    coeff_val = pod.transform(
        trajectories_val
    )

    coeff_test = pod.transform(
        trajectories_test
    )

    # --------------------------------------------------------
    # STATIC SAMPLES
    #
    # condition = (mu, physical time)
    # target    = POD coefficients
    # --------------------------------------------------------

    X_train, Y_train = build_static_samples(
        mus_train,
        times,
        coeff_train,
    )

    X_val, Y_val = build_static_samples(
        mus_val,
        times,
        coeff_val,
    )

    X_test, Y_test = build_static_samples(
        mus_test,
        times,
        coeff_test,
    )

    # --------------------------------------------------------
    # NORMALIZATION
    #
    # Fit ONLY on training data.
    # --------------------------------------------------------

    condition_normalizer = ArrayNormalizer()
    coefficient_normalizer = ArrayNormalizer()

    X_train_norm = (
        condition_normalizer.fit_transform(
            X_train
        )
    )

    Y_train_norm = (
        coefficient_normalizer.fit_transform(
            Y_train
        )
    )

    X_val_norm = (
        condition_normalizer.transform(
            X_val
        )
    )

    X_test_norm = (
        condition_normalizer.transform(
            X_test
        )
    )

    Y_val_norm = (
        coefficient_normalizer.transform(
            Y_val
        )
    )

    Y_test_norm = (
        coefficient_normalizer.transform(
            Y_test
        )
    )

    # --------------------------------------------------------
    # RETURN EVERYTHING NEEDED LATER
    # --------------------------------------------------------

    return {
        "split": split,

        "pod": pod,

        "condition_normalizer":
            condition_normalizer,

        "coefficient_normalizer":
            coefficient_normalizer,

        "trajectories_train":
            trajectories_train,

        "trajectories_val":
            trajectories_val,

        "trajectories_test":
            trajectories_test,

        "mus_train":
            mus_train,

        "mus_val":
            mus_val,

        "mus_test":
            mus_test,

        "coeff_train":
            coeff_train,

        "coeff_val":
            coeff_val,

        "coeff_test":
            coeff_test,

        "X_train":
            X_train,

        "X_val":
            X_val,

        "X_test":
            X_test,

        "Y_train":
            Y_train,

        "Y_val":
            Y_val,

        "Y_test":
            Y_test,

        "X_train_norm":
            X_train_norm,

        "X_val_norm":
            X_val_norm,

        "X_test_norm":
            X_test_norm,

        "Y_train_norm":
            Y_train_norm,

        "Y_val_norm":
            Y_val_norm,

        "Y_test_norm":
            Y_test_norm,
    }


# ============================================================
# TRAIN POD + CONDITIONAL REALNVP
# ============================================================

def train_static_realnvp(
    prepared_data,
    model_config=None,
    train_config=None,
    device=None,
    verbose=True,
):
    """
    Create and train a Conditional RealNVP using data
    returned by prepare_static_data().
    """

    if model_config is None:
        model_config = {
            "n_layers": 8,
            "hidden_size": 128,
            "hidden_depth": 2,
        }

    dim_cond = (
        prepared_data[
            "X_train_norm"
        ].shape[-1]
    )

    dim_y = (
        prepared_data[
            "Y_train_norm"
        ].shape[-1]
    )

    model = ConditionalRealNVP(
        dim_cond=dim_cond,
        dim_y=dim_y,
        **model_config,
    )

    (
        model,
        train_history,
        val_history,
        used_config,
    ) = train_realnvp(
        model=model,

        train_conditions=prepared_data[
            "X_train_norm"
        ],

        train_targets=prepared_data[
            "Y_train_norm"
        ],

        val_conditions=prepared_data[
            "X_val_norm"
        ],

        val_targets=prepared_data[
            "Y_val_norm"
        ],

        config=train_config,
        device=device,
        verbose=verbose,
    )

    return {
        "model": model,

        "train_history":
            train_history,

        "val_history":
            val_history,

        "train_config":
            used_config,

        "model_config":
            model_config,
    }