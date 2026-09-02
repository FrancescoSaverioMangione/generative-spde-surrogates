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
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": None,
    "min_delta": 0.0,
    "restore_best": True,
}


def train_realnvp(
    model,
    X_train,
    Y_train,
    X_val,
    Y_val,
    train_config=None,
    device=None,
    verbose=True,
):
    """
    Train a conditional RealNVP using negative
    log-likelihood.

    Supports optional early stopping based on
    validation loss.
    """

    config = DEFAULT_TRAIN_CONFIG.copy()

    if train_config is not None:
        config.update(
            train_config
        )

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(
        device
    )

    # ========================================================
    # DATA
    # ========================================================

    X_train_tensor = torch.tensor(
        np.asarray(X_train),
        dtype=torch.float32,
    )

    Y_train_tensor = torch.tensor(
        np.asarray(Y_train),
        dtype=torch.float32,
    )

    X_val_tensor = torch.tensor(
        np.asarray(X_val),
        dtype=torch.float32,
        device=device,
    )

    Y_val_tensor = torch.tensor(
        np.asarray(Y_val),
        dtype=torch.float32,
        device=device,
    )

    train_dataset = TensorDataset(
        X_train_tensor,
        Y_train_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(
            config["batch_size"]
        ),
        shuffle=True,
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(
            config["lr"]
        ),
        weight_decay=float(
            config["weight_decay"]
        ),
    )

    # ========================================================
    # EARLY STOPPING STATE
    # ========================================================

    patience = config.get(
        "patience"
    )

    min_delta = float(
        config.get(
            "min_delta",
            0.0,
        )
    )

    restore_best = bool(
        config.get(
            "restore_best",
            True,
        )
    )

    best_val_loss = float(
        "inf"
    )

    best_epoch = None
    best_state = None
    epochs_without_improvement = 0

    train_history = []
    val_history = []

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    n_epochs = int(
        config["epochs"]
    )

    print_every = max(
        1,
        n_epochs // 10,
    )

    for epoch in range(
        n_epochs
    ):

        model.train()

        running_loss = 0.0
        n_seen = 0

        for X_batch, Y_batch in train_loader:

            X_batch = X_batch.to(
                device
            )

            Y_batch = Y_batch.to(
                device
            )

            optimizer.zero_grad()

            loss = -model.log_prob(
                Y_batch,
                X_batch,
            ).mean()

            loss.backward()

            optimizer.step()

            batch_size = (
                X_batch.shape[0]
            )

            running_loss += (
                float(
                    loss.item()
                )
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

            val_loss = float(
                (
                    -model.log_prob(
                        Y_val_tensor,
                        X_val_tensor,
                    ).mean()
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

            best_val_loss = (
                val_loss
            )

            best_epoch = (
                epoch + 1
            )

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
        # EARLY STOP
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
    # RESTORE BEST VALIDATION MODEL
    # ========================================================

    if (
        restore_best
        and best_state is not None
    ):

        model.load_state_dict(
            best_state
        )

    model.eval()

    used_config = (
        config.copy()
    )

    used_config[
        "best_epoch"
    ] = best_epoch

    used_config[
        "best_val_loss"
    ] = best_val_loss

    used_config[
        "epochs_ran"
    ] = len(
        train_history
    )

    return (
        model,
        train_history,
        val_history,
        used_config,
    )