import copy

import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


DEFAULT_TRAIN_CONFIG = {
    "batch_size": 128,
    "epochs": 300,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "patience": None,
    "min_delta": 0.0,
    "restore_best": True,
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
    Train conditional Flow Matching.

    The interpolation path is

        x_tau = (1 - tau) * x0 + tau * x1

    with target velocity

        v_target = x1 - x0

    where:
        x0 ~ N(0, I)
        x1 = data sample.

    Supports:
    - validation monitoring;
    - early stopping;
    - best-model restoration.
    """

    # ========================================================
    # CONFIGURATION
    # ========================================================

    settings = DEFAULT_TRAIN_CONFIG.copy()

    if config is not None:
        settings.update(config)

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(device)

    # ========================================================
    # DATA
    # ========================================================

    train_conditions_tensor = torch.tensor(
        np.asarray(train_conditions),
        dtype=torch.float32,
    )

    train_targets_tensor = torch.tensor(
        np.asarray(train_targets),
        dtype=torch.float32,
    )

    val_conditions_tensor = torch.tensor(
        np.asarray(val_conditions),
        dtype=torch.float32,
        device=device,
    )

    val_targets_tensor = torch.tensor(
        np.asarray(val_targets),
        dtype=torch.float32,
        device=device,
    )

    train_dataset = TensorDataset(
        train_conditions_tensor,
        train_targets_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            settings["batch_size"]
        ),
        shuffle=True,
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            settings["lr"]
        ),
        weight_decay=float(
            settings["weight_decay"]
        ),
    )

    # ========================================================
    # FIXED VALIDATION RANDOMNESS
    # ========================================================
    #
    # Flow Matching validation would otherwise use a new
    # x0 and a new tau at every epoch.
    #
    # Fixing them once makes the validation loss comparable
    # across epochs and therefore more suitable for
    # early stopping / best-model selection.
    # ========================================================

    val_x0 = torch.randn_like(
        val_targets_tensor
    )

    val_tau = torch.rand(
        (
            val_targets_tensor.shape[0],
            1,
        ),
        dtype=torch.float32,
        device=device,
    )

    val_x_tau = (
        (1.0 - val_tau)
        * val_x0
        +
        val_tau
        * val_targets_tensor
    )

    val_target_velocity = (
        val_targets_tensor
        - val_x0
    )

    # ========================================================
    # EARLY STOPPING STATE
    # ========================================================

    patience = settings.get(
        "patience"
    )

    min_delta = float(
        settings.get(
            "min_delta",
            0.0,
        )
    )

    restore_best = bool(
        settings.get(
            "restore_best",
            True,
        )
    )

    best_val_loss = float("inf")
    best_epoch = None
    best_state = None

    epochs_without_improvement = 0

    train_history = []
    val_history = []

    n_epochs = int(
        settings["epochs"]
    )

    print_every = max(
        1,
        n_epochs // 10,
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(n_epochs):

        model.train()

        running_loss = 0.0
        n_seen = 0

        for (
            condition_batch,
            target_batch,
        ) in train_loader:

            condition_batch = (
                condition_batch.to(device)
            )

            target_batch = (
                target_batch.to(device)
            )

            # x1 is the data sample.
            x1 = target_batch

            # x0 is sampled from the base Gaussian.
            x0 = torch.randn_like(x1)

            # Random interpolation time.
            tau = torch.rand(
                (
                    x1.shape[0],
                    1,
                ),
                dtype=x1.dtype,
                device=device,
            )

            # Linear interpolation path.
            x_tau = (
                (1.0 - tau)
                * x0
                +
                tau
                * x1
            )

            # Exact velocity along the linear path.
            target_velocity = (
                x1
                - x0
            )

            optimizer.zero_grad()

            predicted_velocity = model(
                x_tau,
                condition_batch,
                tau,
            )

            loss = torch.mean(
                (
                    predicted_velocity
                    - target_velocity
                )
                ** 2
            )

            loss.backward()

            optimizer.step()

            batch_size = (
                condition_batch.shape[0]
            )

            running_loss += (
                float(loss.item())
                * batch_size
            )

            n_seen += batch_size

        train_loss = (
            running_loss
            / n_seen
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        with torch.no_grad():

            val_predicted_velocity = model(
                val_x_tau,
                val_conditions_tensor,
                val_tau,
            )

            val_loss = float(
                torch.mean(
                    (
                        val_predicted_velocity
                        - val_target_velocity
                    )
                    ** 2
                ).item()
            )

        train_history.append(
            train_loss
        )

        val_history.append(
            val_loss
        )

        # ====================================================
        # BEST MODEL
        # ====================================================

        improved = (
            val_loss
            <
            best_val_loss
            - min_delta
        )

        if improved:

            best_val_loss = val_loss
            best_epoch = epoch + 1

            best_state = copy.deepcopy(
                model.state_dict()
            )

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        # ====================================================
        # PRINT
        # ====================================================

        if verbose and (
            epoch == 0
            or (epoch + 1) % print_every == 0
            or epoch + 1 == n_epochs
        ):

            print(
                f"Epoch "
                f"{epoch + 1:4d}/"
                f"{n_epochs} | "
                f"train="
                f"{train_loss:.6e} | "
                f"val="
                f"{val_loss:.6e} | "
                f"best="
                f"{best_val_loss:.6e}"
            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            patience is not None
            and epochs_without_improvement
            >= int(patience)
        ):

            if verbose:
                print(
                    "Early stopping at epoch "
                    f"{epoch + 1}. "
                    "Best epoch: "
                    f"{best_epoch}."
                )

            break

    # ========================================================
    # RESTORE BEST MODEL
    # ========================================================

    if (
        restore_best
        and best_state is not None
    ):

        model.load_state_dict(
            best_state
        )

    model.eval()

    # ========================================================
    # TRAINING INFORMATION
    # ========================================================

    used_config = settings.copy()

    used_config["best_epoch"] = (
        best_epoch
    )

    used_config["best_val_loss"] = (
        float(best_val_loss)
    )

    used_config["epochs_ran"] = (
        len(train_history)
    )

    return (
        model,
        train_history,
        val_history,
        used_config,
    )