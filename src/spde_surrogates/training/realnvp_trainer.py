import numpy as np
import torch

from torch.utils.data import (
    TensorDataset,
    DataLoader,
)


DEFAULT_TRAIN_CONFIG = {
    "batch_size": 128,
    "epochs": 300,
    "lr": 1e-3,
    "weight_decay": 1e-5,
}


def train_realnvp(
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
    Train a Conditional RealNVP by minimizing the
    Negative Log-Likelihood.

    Parameters
    ----------
    model
        Conditional RealNVP model.

    train_conditions
        Training conditions, shape:
        (n_train_samples, dim_cond)

    train_targets
        Normalized training POD coefficients, shape:
        (n_train_samples, dim_y)

    val_conditions
        Validation conditions.

    val_targets
        Validation POD coefficients.

    config
        Dictionary containing training hyperparameters.

    device
        "cuda" or "cpu".
        If None, CUDA is automatically selected when available.

    verbose
        Print training progress.

    Returns
    -------
    model
        Trained model.

    train_history
        Mean training NLL for every epoch.

    val_history
        Mean validation NLL for every epoch.

    used_config
        Training configuration actually used.
    """

    # --------------------------------------------------------
    # CONFIGURATION
    # --------------------------------------------------------

    used_config = DEFAULT_TRAIN_CONFIG.copy()

    if config is not None:
        used_config.update(config)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(device)

    # --------------------------------------------------------
    # CONVERT NUMPY ARRAYS TO TORCH TENSORS
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

    # --------------------------------------------------------
    # DATALOADERS
    # --------------------------------------------------------

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
    # TRAINING LOOP
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

            target_batch = target_batch.to(
                device
            )

            # Negative Log-Likelihood
            loss = -model.log_prob(
                target_batch,
                cond_batch,
            ).mean()

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

                cond_batch = cond_batch.to(
                    device
                )

                target_batch = target_batch.to(
                    device
                )

                val_loss = -model.log_prob(
                    target_batch,
                    cond_batch,
                ).mean()

                val_losses.append(
                    val_loss.item()
                )

        train_nll = float(
            np.mean(batch_losses)
        )

        val_nll = float(
            np.mean(val_losses)
        )

        train_history.append(
            train_nll
        )

        val_history.append(
            val_nll
        )

        # Print approximately 10 updates during training.
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
                f"train NLL="
                f"{train_nll:.4f} - "
                f"val NLL="
                f"{val_nll:.4f}"
            )

    model.eval()

    return (
        model,
        train_history,
        val_history,
        used_config,
    )