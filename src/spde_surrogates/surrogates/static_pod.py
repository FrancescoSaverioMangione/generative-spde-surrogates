import numpy as np
import torch


class StaticPODGenerativeSurrogate:
    """
    Common wrapper for static POD-based generative models.

    It combines:

        condition normalization
            ->
        conditional generative model
            ->
        coefficient denormalization
            ->
        POD reconstruction

    The underlying generator can be, for example:

        - Conditional RealNVP
        - Conditional Flow Matching
    """

    def __init__(
        self,
        model,
        pod,
        condition_normalizer,
        coefficient_normalizer,
        sampling_config=None,
        device=None,
    ):

        self.model = model
        self.pod = pod

        self.condition_normalizer = (
            condition_normalizer
        )

        self.coefficient_normalizer = (
            coefficient_normalizer
        )

        if sampling_config is None:
            sampling_config = {}

        self.sampling_config = (
            sampling_config.copy()
        )

        if device is None:

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = device

        self.model = self.model.to(
            self.device
        )

        self.model.eval()


    # ========================================================
    # BUILD PHYSICAL CONDITIONS
    # ========================================================

    def _build_conditions(
        self,
        mu,
        times,
    ):

        mu = np.asarray(
            mu,
            dtype=np.float64,
        )

        times = np.asarray(
            times,
            dtype=np.float64,
        )

        if mu.ndim != 1:
            raise ValueError(
                "mu must be a one-dimensional vector."
            )

        if times.ndim != 1:
            raise ValueError(
                "times must be a one-dimensional array."
            )

        mu_grid = np.repeat(
            mu[None, :],
            len(times),
            axis=0,
        )

        conditions = np.concatenate(
            [
                mu_grid,
                times[:, None],
            ],
            axis=1,
        )

        conditions_norm = (
            self.condition_normalizer.transform(
                conditions
            )
        )

        return conditions_norm


    # ========================================================
    # SAMPLE POD COEFFICIENTS
    # ========================================================

    def sample_coefficients(
        self,
        mu,
        times,
        n_samples=1,
        temporal_coupling="shared_noise",
    ):
        """
        Generate trajectories in POD coefficient space.

        Parameters
        ----------
        mu
            Conditioning parameter vector.

        times
            Physical time grid.

        n_samples
            Number of stochastic trajectories to generate.

        temporal_coupling
            "shared_noise":
                the same latent noise is used along all
                physical times of a generated trajectory.

            "independent":
                independent noise is used at every time frame.

        Returns
        -------
        coefficients
            Shape:
                (n_samples, n_times, n_modes)
        """

        conditions_single = (
            self._build_conditions(
                mu,
                times,
            )
        )

        n_times = len(times)

        # Order:
        #
        # trajectory 1:
        #     t0, t1, ..., tN
        #
        # trajectory 2:
        #     t0, t1, ..., tN
        #
        # etc.
        conditions = np.tile(
            conditions_single,
            (
                n_samples,
                1,
            ),
        )

        conditions = torch.as_tensor(
            conditions,
            dtype=torch.float32,
            device=self.device,
        )

        dim_y = self.model.dim_y

        # ----------------------------------------------------
        # TEMPORAL NOISE COUPLING
        # ----------------------------------------------------

        if temporal_coupling == "shared_noise":

            z_trajectory = torch.randn(
                n_samples,
                dim_y,
                device=self.device,
            )

            z = torch.repeat_interleave(
                z_trajectory,
                repeats=n_times,
                dim=0,
            )

        elif temporal_coupling == "independent":

            z = torch.randn(
                n_samples * n_times,
                dim_y,
                device=self.device,
            )

        else:

            raise ValueError(
                "temporal_coupling must be "
                "'shared_noise' or 'independent'."
            )

        # ----------------------------------------------------
        # GENERATIVE MODEL
        # ----------------------------------------------------

        with torch.no_grad():

            generated_norm = (
                self.model.sample(
                    conditions,
                    z=z,
                    **self.sampling_config,
                )
            )

        generated_norm = (
            generated_norm
            .detach()
            .cpu()
            .numpy()
        )

        # ----------------------------------------------------
        # DENORMALIZE POD COEFFICIENTS
        # ----------------------------------------------------

        generated_coefficients = (
            self.coefficient_normalizer
            .inverse_transform(
                generated_norm
            )
        )

        generated_coefficients = (
            generated_coefficients.reshape(
                n_samples,
                n_times,
                dim_y,
            )
        )

        return generated_coefficients


    # ========================================================
    # GENERATE PHYSICAL TRAJECTORIES
    # ========================================================

    def sample(
        self,
        mu,
        times,
        n_samples=1,
        temporal_coupling="shared_noise",
    ):
        """
        Generate trajectories directly in physical space.

        Returns
        -------
        trajectories
            Shape:
                (n_samples, n_times, n_dofs)
        """

        coefficients = (
            self.sample_coefficients(
                mu=mu,
                times=times,
                n_samples=n_samples,
                temporal_coupling=
                    temporal_coupling,
            )
        )

        trajectories = (
            self.pod.inverse_transform(
                coefficients
            )
        )

        return trajectories