import torch
import numpy as np

from spde_surrogates.models.realnvp import (
    ConditionalRealNVP,
)

from spde_surrogates.models.flow_matching import (
    ConditionalPODFlowMatching,
)

from spde_surrogates.rom.pod import (
    PODReducer,
)

from spde_surrogates.data.preprocessing import (
    ArrayNormalizer,
)

from spde_surrogates.surrogates.static_pod import (
    StaticPODGenerativeSurrogate,
)


def _array_to_tensor(array):
    """
    Convert a NumPy array to a CPU torch tensor
    for robust checkpoint serialization.
    """

    return torch.as_tensor(
        np.asarray(array)
    ).cpu()


def _tensor_to_numpy(tensor):
    """
    Convert a checkpoint tensor back to NumPy.
    """

    return tensor.detach().cpu().numpy()


def save_static_surrogate(
    path,
    surrogate,
    model_type,
    model_config,
    train_config=None,
    metadata=None,
):
    """
    Save a complete static POD generative surrogate.

    Parameters
    ----------
    path
        Destination .pt file.

    surrogate
        StaticPODGenerativeSurrogate.

    model_type
        "realnvp" or "flow_matching".

    model_config
        Configuration needed to reconstruct the
        neural network architecture.

    train_config
        Optional training configuration.

    metadata
        Optional additional information.
    """

    if model_type not in {
        "realnvp",
        "flow_matching",
    }:
        raise ValueError(
            "model_type must be "
            "'realnvp' or 'flow_matching'."
        )

    pod = surrogate.pod

    condition_normalizer = (
        surrogate.condition_normalizer
    )

    coefficient_normalizer = (
        surrogate.coefficient_normalizer
    )

    checkpoint = {
        "format_version": 1,

        "model_type":
            model_type,

        "model_config":
            model_config,

        "model_state_dict":
            surrogate.model.state_dict(),

        "sampling_config":
            surrogate.sampling_config,

        "train_config":
            train_config,

        "metadata":
            metadata,

        # --------------------------------------------
        # POD
        # --------------------------------------------

        "pod": {
            "mean":
                _array_to_tensor(
                    pod.mean_
                ),

            "basis":
                _array_to_tensor(
                    pod.basis_
                ),

            "singular_values":
                _array_to_tensor(
                    pod.singular_values_
                ),

            "cumulative_energy":
                _array_to_tensor(
                    pod.cumulative_energy_
                ),

            "n_modes":
                int(
                    pod.n_modes_
                ),

            "n_dofs":
                int(
                    pod.n_dofs_
                ),

            "target_error":
                float(
                    pod.target_error
                ),

            "max_modes":
                int(
                    pod.max_modes
                ),

            "selection":
                pod.selection,

            "energy_threshold":
                float(
                    pod.energy_threshold
                ),
        },

        # --------------------------------------------
        # CONDITION NORMALIZER
        # --------------------------------------------

        "condition_normalizer": {
            "mean":
                _array_to_tensor(
                    condition_normalizer.mean_
                ),

            "std":
                _array_to_tensor(
                    condition_normalizer.std_
                ),

            "eps":
                float(
                    condition_normalizer.eps
                ),
        },

        # --------------------------------------------
        # POD COEFFICIENT NORMALIZER
        # --------------------------------------------

        "coefficient_normalizer": {
            "mean":
                _array_to_tensor(
                    coefficient_normalizer.mean_
                ),

            "std":
                _array_to_tensor(
                    coefficient_normalizer.std_
                ),

            "eps":
                float(
                    coefficient_normalizer.eps
                ),
        },
    }

    torch.save(
        checkpoint,
        path,
    )


def load_static_surrogate(
    path,
    device=None,
):
    """
    Load a complete static POD generative surrogate.

    Returns
    -------
    surrogate
        Restored StaticPODGenerativeSurrogate.

    info
        Dictionary containing model/training metadata.
    """

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True,
    )

    model_type = checkpoint[
        "model_type"
    ]

    model_config = checkpoint[
        "model_config"
    ]

    pod_state = checkpoint[
        "pod"
    ]

    # ========================================================
    # RESTORE POD
    # ========================================================

    pod = PODReducer(
        target_error=pod_state[
            "target_error"
        ],

        max_modes=pod_state[
            "max_modes"
        ],

        selection=pod_state[
            "selection"
        ],

        energy_threshold=pod_state[
            "energy_threshold"
        ],
    )

    pod.mean_ = _tensor_to_numpy(
        pod_state["mean"]
    )

    pod.basis_ = _tensor_to_numpy(
        pod_state["basis"]
    )

    pod.singular_values_ = (
        _tensor_to_numpy(
            pod_state[
                "singular_values"
            ]
        )
    )

    pod.cumulative_energy_ = (
        _tensor_to_numpy(
            pod_state[
                "cumulative_energy"
            ]
        )
    )

    pod.n_modes_ = int(
        pod_state["n_modes"]
    )

    pod.n_dofs_ = int(
        pod_state["n_dofs"]
    )

    pod.is_fitted = True

    # ========================================================
    # RESTORE NORMALIZERS
    # ========================================================

    cond_state = checkpoint[
        "condition_normalizer"
    ]

    condition_normalizer = (
        ArrayNormalizer(
            eps=cond_state["eps"]
        )
    )

    condition_normalizer.mean_ = (
        _tensor_to_numpy(
            cond_state["mean"]
        )
    )

    condition_normalizer.std_ = (
        _tensor_to_numpy(
            cond_state["std"]
        )
    )

    condition_normalizer.is_fitted = True


    coeff_state = checkpoint[
        "coefficient_normalizer"
    ]

    coefficient_normalizer = (
        ArrayNormalizer(
            eps=coeff_state["eps"]
        )
    )

    coefficient_normalizer.mean_ = (
        _tensor_to_numpy(
            coeff_state["mean"]
        )
    )

    coefficient_normalizer.std_ = (
        _tensor_to_numpy(
            coeff_state["std"]
        )
    )

    coefficient_normalizer.is_fitted = True

    # ========================================================
    # RESTORE GENERATIVE MODEL
    # ========================================================

    dim_y = pod.n_modes_

    dim_cond = int(
        condition_normalizer
        .mean_
        .shape[-1]
    )

    if model_type == "realnvp":

        model = ConditionalRealNVP(
            dim_cond=dim_cond,
            dim_y=dim_y,
            **model_config,
        )

    elif model_type == "flow_matching":

        model = (
            ConditionalPODFlowMatching(
                dim_y=dim_y,
                dim_cond=dim_cond,
                **model_config,
            )
        )

    else:

        raise ValueError(
            f"Unknown model type: "
            f"{model_type}"
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(
        device
    )

    model.eval()

    # ========================================================
    # REBUILD COMPLETE SURROGATE
    # ========================================================

    surrogate = (
        StaticPODGenerativeSurrogate(
            model=model,

            pod=pod,

            condition_normalizer=
                condition_normalizer,

            coefficient_normalizer=
                coefficient_normalizer,

            sampling_config=
                checkpoint[
                    "sampling_config"
                ],

            device=device,
        )
    )

    info = {
        "format_version":
            checkpoint[
                "format_version"
            ],

        "model_type":
            model_type,

        "model_config":
            model_config,

        "train_config":
            checkpoint[
                "train_config"
            ],

        "sampling_config":
            checkpoint[
                "sampling_config"
            ],

        "metadata":
            checkpoint[
                "metadata"
            ],
    }

    return surrogate, info