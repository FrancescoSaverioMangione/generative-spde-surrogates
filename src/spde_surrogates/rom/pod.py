import numpy as np


class PODReducer:
    """
    Proper Orthogonal Decomposition reducer for time-dependent trajectories.

    Expected trajectory shape:
        (n_trajectories, n_times, n_dofs)

    The POD is computed by:
        1. flattening all trajectories into snapshots;
        2. subtracting the mean field;
        3. computing the SVD;
        4. selecting a reduced number of modes.

    By default, the number of modes is selected using the
    trajectory-wise relative reconstruction error used in the
    original thesis notebooks.
    """

    def __init__(
        self,
        target_error=1e-2,
        max_modes=100,
        selection="trajectory_error",
        energy_threshold=0.999,
    ):
        self.target_error = target_error
        self.max_modes = max_modes
        self.selection = selection
        self.energy_threshold = energy_threshold

        self.is_fitted = False


    # ========================================================
    # FIT POD
    # ========================================================

    def fit(self, trajectories):
        """
        Fit the POD basis to a set of trajectories.
        """

        trajectories = np.asarray(trajectories)

        if trajectories.ndim != 3:
            raise ValueError(
                "trajectories must have shape "
                "(n_trajectories, n_times, n_dofs)"
            )

        self.n_trajectories_, self.n_times_, self.n_dofs_ = (
            trajectories.shape
        )

        # ----------------------------------------------------
        # Snapshot matrix
        # ----------------------------------------------------

        snapshots = trajectories.reshape(
            self.n_trajectories_ * self.n_times_,
            self.n_dofs_,
        )

        # Mean field
        self.mean_ = snapshots.mean(
            axis=0,
            keepdims=True,
        )

        # Centered snapshots
        snapshots_centered = (
            snapshots - self.mean_
        )

        # ----------------------------------------------------
        # SVD
        # ----------------------------------------------------

        _, singular_values, Vt = np.linalg.svd(
            snapshots_centered,
            full_matrices=False,
        )

        self.singular_values_ = singular_values
        self.full_basis_ = Vt

        self.cumulative_energy_ = (
            np.cumsum(singular_values ** 2)
            / np.sum(singular_values ** 2)
        )

        # We cannot test more modes than actually exist.
        n_test = min(
            self.max_modes,
            Vt.shape[0],
        )

        # ----------------------------------------------------
        # POD diagnostics for different numbers of modes
        # ----------------------------------------------------

        snapshot_errors = []
        trajectory_errors = []
        global_errors = []
        absolute_errors = []
        energy_values = []

        for n in range(1, n_test + 1):

            basis_n = Vt[:n]

            # Projection onto POD space
            coefficients = (
                snapshots_centered
                @ basis_n.T
            )

            # Reconstruction
            snapshots_rec = (
                self.mean_
                + coefficients @ basis_n
            )

            trajectories_rec = (
                snapshots_rec.reshape(
                    self.n_trajectories_,
                    self.n_times_,
                    self.n_dofs_,
                )
            )

            # Error at every time frame.
            num = np.linalg.norm(
                trajectories - trajectories_rec,
                axis=-1,
            )

            den = np.linalg.norm(
                trajectories,
                axis=-1,
            )

            # Legacy Error 1:
            # average relative error snapshot by snapshot.
            snapshot_error = np.mean(
                num / (den + 1e-12)
            )

            # Legacy Error 2:
            # relative error aggregated along each trajectory,
            # then averaged over trajectories.
            trajectory_error = np.mean(
                np.sum(num, axis=-1)
                / (
                    np.sum(den, axis=-1)
                    + 1e-12
                )
            )

            # Global relative Frobenius error.
            global_error = (
                np.linalg.norm(
                    snapshots - snapshots_rec
                )
                / (
                    np.linalg.norm(snapshots)
                    + 1e-12
                )
            )

            # Global absolute Frobenius error.
            absolute_error = np.linalg.norm(
                snapshots - snapshots_rec
            )

            energy_n = (
                np.sum(
                    singular_values[:n] ** 2
                )
                / np.sum(
                    singular_values ** 2
                )
            )

            snapshot_errors.append(
                snapshot_error
            )

            trajectory_errors.append(
                trajectory_error
            )

            global_errors.append(
                global_error
            )

            absolute_errors.append(
                absolute_error
            )

            energy_values.append(
                energy_n
            )

        self.snapshot_errors_ = np.array(
            snapshot_errors
        )

        self.trajectory_errors_ = np.array(
            trajectory_errors
        )

        self.global_errors_ = np.array(
            global_errors
        )

        self.absolute_errors_ = np.array(
            absolute_errors
        )

        self.energy_values_ = np.array(
            energy_values
        )

        # ----------------------------------------------------
        # Select number of POD modes
        # ----------------------------------------------------

        if self.selection == "trajectory_error":

            valid = np.where(
                self.trajectory_errors_
                <= self.target_error
            )[0]

            if len(valid) > 0:
                self.n_modes_ = int(
                    valid[0] + 1
                )
            else:
                self.n_modes_ = n_test

        elif self.selection == "energy":

            valid = np.where(
                self.energy_values_
                >= self.energy_threshold
            )[0]

            if len(valid) > 0:
                self.n_modes_ = int(
                    valid[0] + 1
                )
            else:
                self.n_modes_ = n_test

        else:
            raise ValueError(
                "selection must be "
                "'trajectory_error' or 'energy'"
            )

        # Final POD basis.
        #
        # Shape:
        #     (n_modes, n_dofs)
        self.basis_ = Vt[
            :self.n_modes_
        ]

        self.is_fitted = True

        return self


    # ========================================================
    # PROJECT INTO POD SPACE
    # ========================================================

    def transform(self, data):
        """
        Project snapshots or trajectories onto the POD basis.

        Accepted shapes:
            (n_snapshots, n_dofs)
            (n_trajectories, n_times, n_dofs)
        """

        self._check_fitted()

        data = np.asarray(data)

        if data.ndim == 2:

            centered = (
                data - self.mean_
            )

            return (
                centered
                @ self.basis_.T
            )

        if data.ndim == 3:

            n_traj, n_times, n_dofs = (
                data.shape
            )

            if n_dofs != self.n_dofs_:
                raise ValueError(
                    "Wrong number of spatial DOFs."
                )

            snapshots = data.reshape(
                n_traj * n_times,
                n_dofs,
            )

            coefficients = (
                snapshots - self.mean_
            ) @ self.basis_.T

            return coefficients.reshape(
                n_traj,
                n_times,
                self.n_modes_,
            )

        raise ValueError(
            "data must be a 2D snapshot matrix "
            "or a 3D trajectory tensor"
        )


    # ========================================================
    # RECONSTRUCT FROM POD COEFFICIENTS
    # ========================================================

    def inverse_transform(self, coefficients):
        """
        Reconstruct physical snapshots from POD coefficients.
        """

        self._check_fitted()

        coefficients = np.asarray(
            coefficients
        )

        if coefficients.ndim == 2:

            return (
                self.mean_
                + coefficients
                @ self.basis_
            )

        if coefficients.ndim == 3:

            n_traj, n_times, n_modes = (
                coefficients.shape
            )

            if n_modes != self.n_modes_:
                raise ValueError(
                    "Wrong number of POD modes."
                )

            coeff_flat = coefficients.reshape(
                n_traj * n_times,
                n_modes,
            )

            snapshots = (
                self.mean_
                + coeff_flat
                @ self.basis_
            )

            return snapshots.reshape(
                n_traj,
                n_times,
                self.n_dofs_,
            )

        raise ValueError(
            "coefficients must be 2D or 3D"
        )


    # ========================================================
    # DIRECT RECONSTRUCTION
    # ========================================================

    def reconstruct(self, trajectories):
        """
        Project and immediately reconstruct trajectories.
        """

        coefficients = self.transform(
            trajectories
        )

        return self.inverse_transform(
            coefficients
        )


    # ========================================================
    # FINAL RECONSTRUCTION METRICS
    # ========================================================

    def reconstruction_metrics(self, trajectories):
        """
        Compute POD accuracy and compression metrics.
        """

        self._check_fitted()

        trajectories = np.asarray(
            trajectories
        )

        reconstructed = self.reconstruct(
            trajectories
        )

        difference = (
            trajectories - reconstructed
        )

        absolute_error = np.linalg.norm(
            difference
        )

        relative_error = (
            absolute_error
            / (
                np.linalg.norm(trajectories)
                + 1e-12
            )
        )

        # Error aggregated trajectory by trajectory.
        num = np.linalg.norm(
            difference,
            axis=-1,
        )

        den = np.linalg.norm(
            trajectories,
            axis=-1,
        )

        trajectory_relative_error = np.mean(
            np.sum(num, axis=-1)
            / (
                np.sum(den, axis=-1)
                + 1e-12
            )
        )

        # Relative reconstruction error at each physical time.
        timewise_relative_error = []

        for t in range(
            trajectories.shape[1]
        ):

            err_t = np.linalg.norm(
                difference[:, t, :]
            )

            den_t = np.linalg.norm(
                trajectories[:, t, :]
            )

            if den_t < 1e-12:
                timewise_relative_error.append(
                    np.nan
                )
            else:
                timewise_relative_error.append(
                    err_t / den_t
                )

        return {
            "absolute_error":
                float(absolute_error),

            "relative_error":
                float(relative_error),

            "trajectory_relative_error":
                float(
                    trajectory_relative_error
                ),

            "timewise_relative_error":
                np.array(
                    timewise_relative_error
                ),

            "n_dofs":
                int(self.n_dofs_),

            "n_modes":
                int(self.n_modes_),

            "compression_factor":
                float(
                    self.n_dofs_
                    / self.n_modes_
                ),

            "reduced_fraction":
                float(
                    self.n_modes_
                    / self.n_dofs_
                ),

            "cumulative_energy":
                float(
                    self.cumulative_energy_[
                        self.n_modes_ - 1
                    ]
                ),
        }


    # ========================================================
    # INTERNAL CHECK
    # ========================================================

    def _check_fitted(self):

        if not self.is_fitted:
            raise RuntimeError(
                "PODReducer must be fitted "
                "before calling this method."
            )