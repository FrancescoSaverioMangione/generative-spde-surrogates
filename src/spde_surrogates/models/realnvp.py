import numpy as np
import torch
from torch import nn


class AffineCoupling(nn.Module):
    """
    Conditional affine coupling layer used by the RealNVP model.
    """

    def __init__(
        self,
        dim_y,
        dim_cond,
        mask,
        hidden_size=128,
        hidden_depth=2,
    ):
        super().__init__()

        self.register_buffer(
            "mask",
            mask,
        )

        layers = [
            nn.Linear(
                dim_y + dim_cond,
                hidden_size,
            ),
            nn.SiLU(),
        ]

        for _ in range(hidden_depth):
            layers += [
                nn.Linear(
                    hidden_size,
                    hidden_size,
                ),
                nn.SiLU(),
            ]

        layers.append(
            nn.Linear(
                hidden_size,
                2 * dim_y,
            )
        )

        self.net = nn.Sequential(
            *layers
        )


    def forward(
        self,
        y,
        cond,
        log_det,
        reverse=False,
    ):
        """
        Apply the affine coupling transformation.

        Parameters
        ----------
        y
            Input vector.

        cond
            Conditioning variables.

        log_det
            Current log-determinant.

        reverse
            False:
                POD coefficients -> latent variable z

            True:
                latent variable z -> POD coefficients
        """

        y_masked = (
            y * self.mask
        )

        h = torch.cat(
            [
                y_masked,
                cond,
            ],
            dim=1,
        )

        shift, log_scale = (
            self.net(h).chunk(
                2,
                dim=1,
            )
        )

        # Bound the scaling transformation
        # for numerical stability.
        log_scale = (
            torch.tanh(log_scale)
            * (1.0 - self.mask)
        )

        shift = (
            shift
            * (1.0 - self.mask)
        )

        if not reverse:

            y = (
                y_masked
                + (1.0 - self.mask)
                * (
                    y
                    * torch.exp(log_scale)
                    + shift
                )
            )

            log_det = (
                log_det
                + log_scale.sum(dim=1)
            )

        else:

            y = (
                y_masked
                + (1.0 - self.mask)
                * (
                    (y - shift)
                    * torch.exp(-log_scale)
                )
            )

            log_det = (
                log_det
                - log_scale.sum(dim=1)
            )

        return y, log_det


class ConditionalRealNVP(nn.Module):
    """
    Conditional Normalizing Flow for POD coefficients.
    """

    def __init__(
        self,
        dim_cond,
        dim_y,
        n_layers=8,
        hidden_size=128,
        hidden_depth=2,
    ):
        super().__init__()

        self.dim_cond = dim_cond
        self.dim_y = dim_y

        masks = []

        for k in range(n_layers):

            mask_np = np.array(
                [
                    (i + k) % 2
                    for i in range(dim_y)
                ],
                dtype=np.float32,
            )

            masks.append(
                torch.tensor(
                    mask_np
                ).view(
                    1,
                    dim_y,
                )
            )

        self.layers = nn.ModuleList(
            [
                AffineCoupling(
                    dim_y=dim_y,
                    dim_cond=dim_cond,
                    mask=mask,
                    hidden_size=hidden_size,
                    hidden_depth=hidden_depth,
                )
                for mask in masks
            ]
        )


    def encode(
        self,
        y,
        cond,
    ):
        """
        Map POD coefficients y to latent variables z.
        """

        log_det = torch.zeros(
            y.shape[0],
            device=y.device,
        )

        z = y

        for layer in self.layers:

            z, log_det = layer(
                z,
                cond,
                log_det,
                reverse=False,
            )

        return z, log_det


    def decode(
        self,
        z,
        cond,
    ):
        """
        Map latent variables z to POD coefficients y.
        """

        log_det = torch.zeros(
            z.shape[0],
            device=z.device,
        )

        y = z

        for layer in reversed(
            self.layers
        ):

            y, log_det = layer(
                y,
                cond,
                log_det,
                reverse=True,
            )

        return y


    def log_prob(
        self,
        y,
        cond,
    ):
        """
        Compute log p(y | condition).
        """

        z, log_det = self.encode(
            y,
            cond,
        )

        log_pz = (
            -0.5
            * z.pow(2).sum(dim=1)
        )

        log_pz = (
            log_pz
            - 0.5
            * self.dim_y
            * np.log(
                2.0 * np.pi
            )
        )

        return (
            log_pz
            + log_det
        )


    @torch.no_grad()
    def sample(
        self,
        cond,
        z=None,
    ):
        """
        Generate conditional POD coefficients.

        If z is not supplied, standard Gaussian noise is sampled.
        """

        if z is None:

            z = torch.randn(
                cond.shape[0],
                self.dim_y,
                device=cond.device,
            )

        return self.decode(
            z,
            cond,
        )