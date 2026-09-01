import numpy as np


def split_by_group(
    group_ids,
    train_fraction=0.75,
    val_fraction=0.15,
    test_fraction=0.10,
    seed=1,
):
    """
    Split trajectories into train, validation and test sets
    keeping all trajectories associated with the same group
    (same conditioning parameter mu) in the same split.

    Parameters
    ----------
    group_ids : array-like
        Group identifier for every trajectory.

    train_fraction : float
        Fraction of distinct groups assigned to training.

    val_fraction : float
        Fraction of distinct groups assigned to validation.

    test_fraction : float
        Fraction of distinct groups assigned to testing.

    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Dictionary containing group IDs and trajectory indices
        for train, validation and test sets.
    """

    group_ids = np.asarray(group_ids)

    if not np.isclose(
        train_fraction
        + val_fraction
        + test_fraction,
        1.0,
    ):
        raise ValueError(
            "train_fraction + val_fraction + "
            "test_fraction must equal 1."
        )

    unique_groups = np.unique(group_ids)

    if len(unique_groups) < 3:
        raise ValueError(
            "At least 3 distinct groups are required "
            "for train/validation/test splitting."
        )

    rng = np.random.default_rng(seed)

    shuffled_groups = rng.permutation(
        unique_groups
    )

    n_groups = len(shuffled_groups)

    n_train = int(
        np.floor(
            train_fraction * n_groups
        )
    )

    n_val = int(
        np.floor(
            val_fraction * n_groups
        )
    )

    # Everything remaining goes to the test set.
    n_test = (
        n_groups
        - n_train
        - n_val
    )

    if (
        n_train == 0
        or n_val == 0
        or n_test == 0
    ):
        raise ValueError(
            "The selected fractions produce an empty split. "
            "Use more groups or different fractions."
        )

    train_groups = shuffled_groups[
        :n_train
    ]

    val_groups = shuffled_groups[
        n_train:n_train + n_val
    ]

    test_groups = shuffled_groups[
        n_train + n_val:
    ]

    train_indices = np.where(
        np.isin(
            group_ids,
            train_groups
        )
    )[0]

    val_indices = np.where(
        np.isin(
            group_ids,
            val_groups
        )
    )[0]

    test_indices = np.where(
        np.isin(
            group_ids,
            test_groups
        )
    )[0]

    return {
        "train_groups": train_groups,
        "val_groups": val_groups,
        "test_groups": test_groups,

        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
    }