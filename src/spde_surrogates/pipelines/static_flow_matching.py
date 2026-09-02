from spde_surrogates.models.flow_matching import (
    ConditionalPODFlowMatching,
)

from spde_surrogates.training.flow_matching_trainer import (
    train_flow_matching,
)


def train_static_flow_matching(
    prepared_data,
    model_config=None,
    train_config=None,
    sampling_config=None,
    device=None,
    verbose=True,
):
    """
    Train a static POD + Flow Matching surrogate.

    IMPORTANT:
    prepared_data is the SAME object produced by
    prepare_static_data() and used by the RealNVP pipeline.

    This guarantees that NF and FM use exactly the same:
        - train/validation/test split
        - POD basis
        - POD dimension
        - condition normalization
        - coefficient normalization
    """

    if model_config is None:

        model_config = {
            "hidden_dims":
                (256, 256, 256),
        }

    if sampling_config is None:

        sampling_config = {
            "n_steps": 100,
        }

    dim_cond = (
        prepared_data[
            "X_train_norm"
        ].shape[-1]
    )

    dim_y = (
        prepared_data[
            "Y_train_norm"
        ].shape[-1]
    )

    model = (
        ConditionalPODFlowMatching(
            dim_y=dim_y,
            dim_cond=dim_cond,
            **model_config,
        )
    )

    (
        model,
        train_history,
        val_history,
        used_config,
    ) = train_flow_matching(

        model=model,

        train_conditions=prepared_data[
            "X_train_norm"
        ],

        train_targets=prepared_data[
            "Y_train_norm"
        ],

        val_conditions=prepared_data[
            "X_val_norm"
        ],

        val_targets=prepared_data[
            "Y_val_norm"
        ],

        config=train_config,
        device=device,
        verbose=verbose,
    )

    return {
        "model": model,

        "train_history":
            train_history,

        "val_history":
            val_history,

        "train_config":
            used_config,

        "model_config":
            model_config,

        "sampling_config":
            sampling_config,
    }