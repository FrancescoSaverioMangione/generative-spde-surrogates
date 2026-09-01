import numpy as np


class ArrayNormalizer:
    """
    Standardization of arrays along their last dimension.

    The normalizer computes statistics only during fit().
    Validation and test data must only use transform().
    """

    def __init__(
        self,
        eps=1e-8,
    ):
        self.eps = eps
        self.is_fitted = False


    def fit(self, data):
        """
        Fit mean and standard deviation.

        Any leading dimensions are interpreted as samples.
        The last dimension represents features.
        """

        data = np.asarray(
            data,
            dtype=np.float64,
        )

        if data.ndim < 2:
            raise ValueError(
                "data must have at least 2 dimensions"
            )

        flat = data.reshape(
            -1,
            data.shape[-1],
        )

        self.mean_ = flat.mean(
            axis=0,
            keepdims=True,
        )

        self.std_ = flat.std(
            axis=0,
            keepdims=True,
        )

        # Prevent division by zero for constant features.
        self.std_ = np.where(
            self.std_ < self.eps,
            1.0,
            self.std_,
        )

        self.is_fitted = True

        return self


    def transform(self, data):
        """
        Standardize data using previously fitted statistics.
        """

        self._check_fitted()

        data = np.asarray(
            data,
            dtype=np.float64,
        )

        return (
            data - self.mean_
        ) / self.std_


    def inverse_transform(
        self,
        data,
    ):
        """
        Undo the normalization.
        """

        self._check_fitted()

        data = np.asarray(
            data,
            dtype=np.float64,
        )

        return (
            data * self.std_
            + self.mean_
        )


    def fit_transform(
        self,
        data,
    ):
        """
        Fit the normalizer and immediately transform the data.
        """

        self.fit(data)

        return self.transform(
            data
        )


    def _check_fitted(self):

        if not self.is_fitted:
            raise RuntimeError(
                "ArrayNormalizer must be fitted "
                "before transform or inverse_transform."
            )


# ============================================================
# STATIC GENERATIVE DATASET
# ============================================================

def build_static_samples(
    mus,
    times,
    coefficients,
):
    """
    Convert trajectory-wise POD coefficients into static
    training samples conditioned on (mu, t).

    Parameters
    ----------
    mus
        Shape:
            (n_trajectories, n_parameters)

    times
        Shape:
            (n_times,)

    coefficients
        Shape:
            (n_trajectories, n_times, n_modes)

    Returns
    -------
    conditions
        Shape:
            (n_trajectories * n_times,
             n_parameters + 1)

        Each row contains:
            [mu_1, ..., mu_p, t]

    targets
        Shape:
            (n_trajectories * n_times,
             n_modes)
    """

    mus = np.asarray(mus)
    times = np.asarray(times)
    coefficients = np.asarray(
        coefficients
    )

    n_traj, n_times, n_modes = (
        coefficients.shape
    )

    if mus.shape[0] != n_traj:
        raise ValueError(
            "mus and coefficients must contain "
            "the same number of trajectories."
        )

    if len(times) != n_times:
        raise ValueError(
            "times and coefficients have "
            "incompatible time dimensions."
        )

    # Repeat each mu for every physical time.
    mu_grid = np.repeat(
        mus[:, None, :],
        n_times,
        axis=1,
    )

    # Create physical-time column.
    time_grid = np.broadcast_to(
        times[None, :, None],
        (
            n_traj,
            n_times,
            1,
        ),
    )

    conditions = np.concatenate(
        [
            mu_grid,
            time_grid,
        ],
        axis=-1,
    )

    conditions = conditions.reshape(
        n_traj * n_times,
        -1,
    )

    targets = coefficients.reshape(
        n_traj * n_times,
        n_modes,
    )

    return (
        conditions,
        targets,
    )