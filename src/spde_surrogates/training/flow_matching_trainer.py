import numpy as np
import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)


DEFAULT_FM_TRAIN_CONFIG = {
    "batch_size": 128,
    "epochs": 300,
    "lr": 1e-4,
    "weight_decay": 1e-5,
}


def train_flow_matching(
    model,
    train_conditions,
    train_targets,
    val_conditions,
    val_targets,
    config=None,
    device=None,
    verbose=True,
):
    """
    Train a Conditional Flow Matching model.

    The target velocity is

        v_target = x1 - x0

    along the linear interpolation

        x_tau = (1 - tau) * x0 + tau * x1

    where x0 is Gaussian noise and x1 is a normalized
    POD coefficient vector.
    """

    used_config = (
        DEFAULT_FM_TRAIN_CONFIG.copy()
    )

    if config is not None:
        used_config.update(
            config
        )

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(device)

    # --------------------------------------------------------
    # NUMPY -> TORCH
    # --------------------------------------------------------

    train_conditions = torch.as_tensor(
        train_conditions,
        dtype=torch.float32,
    )

    train_targets = torch.as_tensor(
        train_targets,
        dtype=torch.float32,
    )

    val_conditions = torch.as_tensor(
        val_conditions,
        dtype=torch.float32,
    )

    val_targets = torch.as_tensor(
        val_targets,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        train_conditions,
        train_targets,
    )

    val_dataset = TensorDataset(
        val_conditions,
        val_targets,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(
            used_config["batch_size"],
            len(train_dataset),
        ),
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=min(
            used_config["batch_size"],
            len(val_dataset),
        ),
        shuffle=False,
        drop_last=False,
    )

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=used_config["lr"],
        weight_decay=used_config[
            "weight_decay"
        ],
    )

    train_history = []
    val_history = []

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    for epoch in range(
        used_config["epochs"]
    ):

        model.train()

        batch_losses = []

        for (
            cond_batch,
            target_batch,
        ) in train_loader:

            cond_batch = cond_batch.to(
                device
            )

            # x1 = data sample
            x1 = target_batch.to(
                device
            )

            # x0 = Gaussian source
            x0 = torch.randn_like(
                x1
            )

            # Artificial Flow Matching time
            tau = torch.rand(
                x1.shape[0],
                1,
                device=device,
            )

            # Linear interpolation
            x_tau = (
                (1.0 - tau) * x0
                + tau * x1
            )

            # Exact velocity along this path
            v_target = (
                x1 - x0
            )

            # Predicted velocity
            v_pred = model(
                x_tau,
                cond_batch,
                tau,
            )

            # Flow Matching loss
            loss = (
                v_target
                - v_pred
            ).pow(2).mean()

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            batch_losses.append(
                loss.item()
            )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        model.eval()

        val_losses = []

        with torch.no_grad():

            for (
                cond_batch,
                target_batch,
            ) in val_loader:

                cond_batch = (
                    cond_batch.to(
                        device
                    )
                )

                x1 = target_batch.to(
                    device
                )

                x0 = torch.randn_like(
                    x1
                )

                tau = torch.rand(
                    x1.shape[0],
                    1,
                    device=device,
                )

                x_tau = (
                    (1.0 - tau) * x0
                    + tau * x1
                )

                v_target = (
                    x1 - x0
                )

                v_pred = model(
                    x_tau,
                    cond_batch,
                    tau,
                )

                val_loss = (
                    v_target
                    - v_pred
                ).pow(2).mean()

                val_losses.append(
                    val_loss.item()
                )

        train_loss = float(
            np.mean(
                batch_losses
            )
        )

        val_loss = float(
            np.mean(
                val_losses
            )
        )

        train_history.append(
            train_loss
        )

        val_history.append(
            val_loss
        )

        print_every = max(
            1,
            used_config["epochs"] // 10,
        )

        if (
            verbose
            and (
                epoch + 1
            ) % print_every == 0
        ):

            print(
                f"epoch "
                f"{epoch + 1:4d}/"
                f"{used_config['epochs']} - "
                f"train FM loss="
                f"{train_loss:.4e} - "
                f"val FM loss="
                f"{val_loss:.4e}"
            )

    model.eval()

    return (
        model,
        train_history,
        val_history,
        used_config,
    )