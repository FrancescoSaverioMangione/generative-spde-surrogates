import torch
from torch import nn


class ConditionalPODFlowMatching(nn.Module):
    """
    Conditional Flow Matching model for normalized POD coefficients.

    The model learns the velocity field

        v_theta(x_tau, condition, tau)

    where:
        x_tau      = point along the artificial flow,
        condition  = physical condition (mu, physical time),
        tau        = artificial Flow Matching time.
    """

    def __init__(
        self,
        dim_y,
        dim_cond,
        hidden_dims=(256, 256, 256),
    ):
        super().__init__()

        self.dim_y = dim_y
        self.dim_cond = dim_cond

        widths = (
            [dim_y + dim_cond + 1]
            + list(hidden_dims)
            + [dim_y]
        )

        layers = []

        for i in range(
            len(widths) - 2
        ):
            layers.append(
                nn.Linear(
                    widths[i],
                    widths[i + 1],
                )
            )

            layers.append(
                nn.SiLU()
            )

        layers.append(
            nn.Linear(
                widths[-2],
                widths[-1],
            )
        )

        self.net = nn.Sequential(
            *layers
        )


    def forward(
        self,
        x,
        cond,
        tau,
    ):
        """
        Evaluate the learned velocity field.

        x
            Current normalized POD state.

        cond
            Normalized physical condition (mu, t_phys).

        tau
            Artificial Flow Matching time in [0, 1].
        """

        if tau.ndim == 0:

            tau = tau.view(
                1,
                1,
            ).repeat(
                x.shape[0],
                1,
            )

        elif (
            tau.shape[0] == 1
            and x.shape[0] > 1
        ):

            tau = tau.repeat(
                x.shape[0],
                1,
            )

        h = torch.cat(
            [
                x,
                cond,
                tau,
            ],
            dim=1,
        )

        return self.net(h)


    @torch.no_grad()
    def sample(
        self,
        cond,
        n_steps=100,
        z=None,
    ):
        """
        Generate normalized POD coefficients by integrating
        the learned velocity field from tau=0 to tau=1.

        Euler integration is used, matching the original
        thesis notebook.

        If z is supplied, it is used as the initial Gaussian
        state. Otherwise new Gaussian noise is sampled.
        """

        batch_size = cond.shape[0]
        device = cond.device

        if z is None:

            x = torch.randn(
                batch_size,
                self.dim_y,
                device=device,
            )

        else:

            x = z.to(device)

        taus = torch.linspace(
            0.0,
            1.0,
            n_steps,
            device=device,
        )

        for i in range(
            n_steps - 1
        ):

            tau = taus[i].view(
                1,
                1,
            ).repeat(
                batch_size,
                1,
            )

            dtau = (
                taus[i + 1]
                - taus[i]
            )

            velocity = self.forward(
                x,
                cond,
                tau,
            )

            x = (
                x
                + velocity * dtau
            )

        return x