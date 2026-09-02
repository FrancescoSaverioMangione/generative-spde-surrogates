from importlib import import_module
from pathlib import Path

import numpy as np
import torch

from spde_surrogates.data.dataset import (
    get_fom_dataset,
)

from spde_surrogates.pipelines.static_realnvp import (
    prepare_static_data,
    train_static_realnvp,
)

from spde_surrogates.pipelines.static_flow_matching import (
    train_static_flow_matching,
)

from spde_surrogates.surrogates.static_pod import (
    StaticPODGenerativeSurrogate,
)

from spde_surrogates.evaluation.evaluator import (
    evaluate_surrogate,
)

from spde_surrogates.evaluation.benchmark import (
    benchmark_surrogates,
)

from spde_surrogates.persistence.checkpoint import (
    save_static_surrogate,
)


def _load_problem(config):
    """
    Import the physical problem specified in the YAML configuration.
    """

    module_name = config["problem"]["module"]

    return import_module(module_name)


def _set_random_seed(seed):
    """
    Set NumPy and PyTorch random seeds.
    """

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_static_comparison(
    config,
    device=None,
    verbose=True,
):
    """
    Run the complete static POD generative comparison.

    Pipeline
    --------
    1. Load the physical problem.
    2. Load or generate the FOM dataset.
    3. Split data, fit POD and normalize.
    4. Train RealNVP.
    5. Train Flow Matching.
    6. Evaluate RealNVP.
    7. Evaluate Flow Matching.
    8. Benchmark generation speed.
    9. Save complete checkpoints.

    Parameters
    ----------
    config
        Experiment configuration dictionary.

    device
        Optional PyTorch device.

    verbose
        Print progress information.

    Returns
    -------
    dict
        Important experiment objects and results.
    """

    # ========================================================
    # EXPERIMENT SETTINGS
    # ========================================================

    experiment_config = config["experiment"]
    data_config = config["data"]

    experiment_name = experiment_config["name"]
    seed = int(experiment_config["seed"])

    _set_random_seed(seed)

    output_dir = Path(
        experiment_config["output_dir"]
    )

    checkpoint_dir = (
        output_dir
        / "checkpoints"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_path = Path(
        data_config["path"]
    )

    data_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if verbose:
        print("=" * 70)
        print(f"Experiment: {experiment_name}")
        print("=" * 70)

    # ========================================================
    # 1. PHYSICAL PROBLEM
    # ========================================================

    if verbose:
        print(
            "\n[1/9] Loading physical problem..."
        )

    problem = _load_problem(config)

    if verbose:
        print(
            "Problem module:",
            config["problem"]["module"],
        )

    # ========================================================
    # 2. FOM DATASET
    # ========================================================

    if verbose:
        print(
            "\n[2/9] Loading or generating FOM dataset..."
        )

    (
        mu_bank,
        times,
        mus,
        trajectories,
        group_ids,
    ) = get_fom_dataset(
        path=data_path,
        problem=problem,
        n_mu=int(
            data_config["n_mu"]
        ),
        n_realizations=int(
            data_config["n_realizations"]
        ),
        nimp=int(
            data_config["nimp"]
        ),
        dt=float(
            data_config["dt"]
        ),
        steps=int(
            data_config["steps"]
        ),
        seed=seed,
    )

    if verbose:
        print(
            "mu_bank shape:",
            mu_bank.shape,
        )

        print(
            "times shape:",
            times.shape,
        )

        print(
            "trajectories shape:",
            trajectories.shape,
        )

        print(
            "group_ids shape:",
            group_ids.shape,
        )

    # ========================================================
    # 3. SPLIT + POD + NORMALIZATION
    # ========================================================

    if verbose:
        print(
            "\n[3/9] Preparing common POD representation..."
        )

    prepared = prepare_static_data(
        trajectories=trajectories,
        conds=mus,
        times=times,
        group_ids=group_ids,
        split_config=config["split"],
        pod_config=config["pod"],
    )

    n_times = len(times)

    n_train_static = (
        len(
            prepared["split"]["train_indices"]
        )
        * n_times
    )

    n_val_static = (
        len(
            prepared["split"]["val_indices"]
        )
        * n_times
    )

    n_test_static = (
        len(
            prepared["split"]["test_indices"]
        )
        * n_times
    )

    if verbose:
        print(
            "Selected POD modes:",
            prepared["pod"].n_modes_,
        )

        print(
            "Training static samples:",
            n_train_static,
        )

        print(
            "Validation static samples:",
            n_val_static,
        )

        print(
            "Test static samples:",
            n_test_static,
        )

    # ========================================================
    # 4. REALNVP
    # ========================================================

    if verbose:
        print(
            "\n[4/9] Training RealNVP..."
        )

    nf_result = train_static_realnvp(
        prepared_data=prepared,
        model_config=config[
            "realnvp"
        ]["model"],
        train_config=config[
            "realnvp"
        ]["training"],
        device=device,
        verbose=verbose,
    )

    nf_surrogate = StaticPODGenerativeSurrogate(
        model=nf_result["model"],
        pod=prepared["pod"],
        condition_normalizer=prepared[
            "condition_normalizer"
        ],
        coefficient_normalizer=prepared[
            "coefficient_normalizer"
        ],
        sampling_config={},
        device=device,
    )

    # ========================================================
    # 5. FLOW MATCHING
    # ========================================================

    if verbose:
        print(
            "\n[5/9] Training Flow Matching..."
        )

    fm_result = train_static_flow_matching(
        prepared_data=prepared,
        model_config=config[
            "flow_matching"
        ]["model"],
        train_config=config[
            "flow_matching"
        ]["training"],
        sampling_config=config[
            "flow_matching"
        ]["sampling"],
        device=device,
        verbose=verbose,
    )

    fm_surrogate = StaticPODGenerativeSurrogate(
        model=fm_result["model"],
        pod=prepared["pod"],
        condition_normalizer=prepared[
            "condition_normalizer"
        ],
        coefficient_normalizer=prepared[
            "coefficient_normalizer"
        ],
        sampling_config=config[
            "flow_matching"
        ]["sampling"],
        device=device,
    )

    # ========================================================
    # TEST GROUP IDS
    # ========================================================

    test_indices = prepared[
        "split"
    ]["test_indices"]

    test_group_ids = group_ids[
        test_indices
    ]

    # ========================================================
    # 6. REALNVP EVALUATION
    # ========================================================

    if verbose:
        print(
            "\n[6/9] Evaluating RealNVP..."
        )

    evaluation_config = config[
        "evaluation"
    ]

    nf_evaluation = evaluate_surrogate(
        surrogate=nf_surrogate,
        real_trajectories=prepared[
            "trajectories_test"
        ],
        mus=prepared[
            "mus_test"
        ],
        group_ids=test_group_ids,
        times=times,
        n_generated=int(
            evaluation_config[
                "n_generated"
            ]
        ),
        temporal_coupling=
            evaluation_config[
                "temporal_coupling"
            ],
        verbose=verbose,
    )

    # ========================================================
    # 7. FLOW MATCHING EVALUATION
    # ========================================================

    if verbose:
        print(
            "\n[7/9] Evaluating Flow Matching..."
        )

    fm_evaluation = evaluate_surrogate(
        surrogate=fm_surrogate,
        real_trajectories=prepared[
            "trajectories_test"
        ],
        mus=prepared[
            "mus_test"
        ],
        group_ids=test_group_ids,
        times=times,
        n_generated=int(
            evaluation_config[
                "n_generated"
            ]
        ),
        temporal_coupling=
            evaluation_config[
                "temporal_coupling"
            ],
        verbose=verbose,
    )

    # ========================================================
    # 8. GENERATIVE EFFICIENCY
    # ========================================================

    if verbose:
        print(
            "\n[8/9] Benchmarking generation..."
        )

    benchmark_config = config[
        "benchmark"
    ]

    benchmark_mu = prepared[
        "mus_test"
    ][0]

    benchmark_results = benchmark_surrogates(
        surrogates={
            "NF": nf_surrogate,
            "FM": fm_surrogate,
        },
        mu=benchmark_mu,
        times=times,
        sample_sizes=benchmark_config[
            "sample_sizes"
        ],
        n_repeats=int(
            benchmark_config[
                "n_repeats"
            ]
        ),
        n_warmup=int(
            benchmark_config[
                "n_warmup"
            ]
        ),
        temporal_coupling=
            evaluation_config[
                "temporal_coupling"
            ],
    )

    # ========================================================
    # 9. SAVE CHECKPOINTS
    # ========================================================

    if verbose:
        print(
            "\n[9/9] Saving checkpoints..."
        )

    nf_checkpoint = (
        checkpoint_dir
        / "realnvp.pt"
    )

    fm_checkpoint = (
        checkpoint_dir
        / "flow_matching.pt"
    )

    common_metadata = {
        "experiment_name":
            experiment_name,

        "seed":
            seed,

        "problem_module":
            config["problem"]["module"],

        "n_mu":
            int(
                data_config["n_mu"]
            ),

        "n_realizations":
            int(
                data_config[
                    "n_realizations"
                ]
            ),

        "n_times":
            int(
                len(times)
            ),

        "n_dofs":
            int(
                trajectories.shape[-1]
            ),

        "n_pod_modes":
            int(
                prepared[
                    "pod"
                ].n_modes_
            ),
    }

    save_static_surrogate(
        path=nf_checkpoint,
        surrogate=nf_surrogate,
        model_type="realnvp",
        model_config=config[
            "realnvp"
        ]["model"],
        train_config=config[
            "realnvp"
        ]["training"],
        metadata=common_metadata,
    )

    save_static_surrogate(
        path=fm_checkpoint,
        surrogate=fm_surrogate,
        model_type="flow_matching",
        model_config=config[
            "flow_matching"
        ]["model"],
        train_config=config[
            "flow_matching"
        ]["training"],
        metadata=common_metadata,
    )

    if verbose:
        print(
            "\nCheckpoints:"
        )

        print(
            "  NF:",
            nf_checkpoint,
        )

        print(
            "  FM:",
            fm_checkpoint,
        )

        print(
            "\nExperiment completed."
        )

        print("=" * 70)

    # ========================================================
    # RETURN RESULTS
    # ========================================================

    return {
        "config":
            config,

        "problem":
            problem,

        "mu_bank":
            mu_bank,

        "times":
            times,

        "mus":
            mus,

        "group_ids":
            group_ids,

        "trajectories":
            trajectories,

        "prepared":
            prepared,

        "nf_result":
            nf_result,

        "fm_result":
            fm_result,

        "nf_surrogate":
            nf_surrogate,

        "fm_surrogate":
            fm_surrogate,

        "nf_evaluation":
            nf_evaluation,

        "fm_evaluation":
            fm_evaluation,

        "benchmark":
            benchmark_results,

        "nf_checkpoint":
            str(
                nf_checkpoint
            ),

        "fm_checkpoint":
            str(
                fm_checkpoint
            ),
    }