from pathlib import Path

import csv
import json

import matplotlib.pyplot as plt
import numpy as np


def _write_csv(
    path,
    fieldnames,
    rows,
):
    """
    Write a list of dictionaries to CSV.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def _find_benchmark_result(
    benchmark,
    model_name,
    n_samples,
):
    """
    Find one benchmark row for the requested
    model and number of samples.
    """

    for row in benchmark[model_name]:

        if int(row["n_samples"]) == int(
            n_samples
        ):
            return row

    return None


def export_static_comparison_report(
    output_dir,
    config,
    prepared_data,
    pod_metrics,
    nf_result,
    fm_result,
    nf_evaluation,
    fm_evaluation,
    benchmark,
    times,
):
    """
    Export the main quantitative results of a
    static RealNVP vs Flow Matching comparison.

    Parameters
    ----------
    output_dir
        Destination directory.

    config
        Experiment configuration.

    prepared_data
        Output of prepare_static_data.

    pod_metrics
        Dictionary with keys:
            "train"
            "val"
            "test"

    nf_result
        RealNVP training result.

    fm_result
        Flow Matching training result.

    nf_evaluation
        RealNVP evaluation result.

    fm_evaluation
        Flow Matching evaluation result.

    benchmark
        Common generation benchmark.

    times
        Physical time grid.

    Returns
    -------
    dict
        Paths of exported files.
    """

    output_dir = Path(
        output_dir
    )

    tables_dir = (
        output_dir
        / "tables"
    )

    figures_dir = (
        output_dir
        / "figures"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # BASIC INFORMATION
    # ========================================================

    pod = prepared_data["pod"]

    nf_summary = nf_evaluation[
        "summary"
    ]

    fm_summary = fm_evaluation[
        "summary"
    ]

    nf_train_config = nf_result[
        "train_config"
    ]

    fm_train_config = fm_result[
        "train_config"
    ]

    # Use the largest common benchmark batch.
    nf_sizes = {
        int(row["n_samples"])
        for row in benchmark["RealNVP"]
    }

    fm_sizes = {
        int(row["n_samples"])
        for row in benchmark["Flow Matching"]
    }

    common_sizes = sorted(
        nf_sizes & fm_sizes
    )

    largest_benchmark_size = (
        common_sizes[-1]
        if common_sizes
        else None
    )

    nf_benchmark_large = None
    fm_benchmark_large = None

    if largest_benchmark_size is not None:

        nf_benchmark_large = (
            _find_benchmark_result(
                benchmark,
                "RealNVP",
                largest_benchmark_size,
            )
        )

        fm_benchmark_large = (
            _find_benchmark_result(
                benchmark,
                "Flow Matching",
                largest_benchmark_size,
            )
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "experiment_name":
            config["experiment"]["name"],

        "dataset": {
            "n_mu":
                int(
                    config["data"]["n_mu"]
                ),

            "n_realizations":
                int(
                    config["data"][
                        "n_realizations"
                    ]
                ),

            "n_times":
                int(
                    len(times)
                ),

            "n_dofs":
                int(
                    pod.n_dofs_
                ),
        },

        "pod": {
            "n_modes":
                int(
                    pod.n_modes_
                ),

            "compression_factor":
                float(
                    pod.n_dofs_
                    / pod.n_modes_
                ),

            "train_trajectory_relative_error":
                float(
                    pod_metrics[
                        "train"
                    ][
                        "trajectory_relative_error"
                    ]
                ),

            "val_trajectory_relative_error":
                float(
                    pod_metrics[
                        "val"
                    ][
                        "trajectory_relative_error"
                    ]
                ),

            "test_trajectory_relative_error":
                float(
                    pod_metrics[
                        "test"
                    ][
                        "trajectory_relative_error"
                    ]
                ),
        },

        "realnvp": {
            "energy_mean":
                float(
                    nf_summary["energy_mean"]
                ),

            "energy_median":
                float(
                    nf_summary[
                        "energy_median"
                    ]
                ),

            "energy_min":
                float(
                    nf_summary["energy_min"]
                ),

            "energy_max":
                float(
                    nf_summary["energy_max"]
                ),

            "energy_std":
                float(
                    nf_summary["energy_std"]
                ),

            "nearest_neighbor_mean":
                float(
                    nf_summary[
                        "nearest_neighbor_mean"
                    ]
                ),

            "best_epoch":
                int(
                    nf_train_config[
                        "best_epoch"
                    ]
                ),

            "best_val_loss":
                float(
                    nf_train_config[
                        "best_val_loss"
                    ]
                ),

            "epochs_ran":
                int(
                    nf_train_config[
                        "epochs_ran"
                    ]
                ),
        },

        "flow_matching": {
            "energy_mean":
                float(
                    fm_summary["energy_mean"]
                ),

            "energy_median":
                float(
                    fm_summary[
                        "energy_median"
                    ]
                ),

            "energy_min":
                float(
                    fm_summary["energy_min"]
                ),

            "energy_max":
                float(
                    fm_summary["energy_max"]
                ),

            "energy_std":
                float(
                    fm_summary["energy_std"]
                ),

            "nearest_neighbor_mean":
                float(
                    fm_summary[
                        "nearest_neighbor_mean"
                    ]
                ),

            "best_epoch":
                int(
                    fm_train_config[
                        "best_epoch"
                    ]
                ),

            "best_val_loss":
                float(
                    fm_train_config[
                        "best_val_loss"
                    ]
                ),

            "epochs_ran":
                int(
                    fm_train_config[
                        "epochs_ran"
                    ]
                ),
        },
    }

    # ========================================================
    # COMPARATIVE QUANTITIES
    # ========================================================

    summary[
        "comparison"
    ] = {
        "fm_energy_mean_reduction_percent":
            float(
                100.0
                * (
                    nf_summary["energy_mean"]
                    - fm_summary["energy_mean"]
                )
                / nf_summary["energy_mean"]
            ),

        "fm_nn_reduction_percent":
            float(
                100.0
                * (
                    nf_summary[
                        "nearest_neighbor_mean"
                    ]
                    - fm_summary[
                        "nearest_neighbor_mean"
                    ]
                )
                / nf_summary[
                    "nearest_neighbor_mean"
                ]
            ),
    }

    if (
        nf_benchmark_large is not None
        and fm_benchmark_large is not None
    ):

        summary[
            "comparison"
        ][
            "benchmark_n_samples"
        ] = int(
            largest_benchmark_size
        )

        summary[
            "comparison"
        ][
            "realnvp_median_seconds"
        ] = float(
            nf_benchmark_large[
                "median_seconds"
            ]
        )

        summary[
            "comparison"
        ][
            "flow_matching_median_seconds"
        ] = float(
            fm_benchmark_large[
                "median_seconds"
            ]
        )

        summary[
            "comparison"
        ][
            "flow_matching_slowdown"
        ] = float(
            fm_benchmark_large[
                "median_seconds"
            ]
            /
            nf_benchmark_large[
                "median_seconds"
            ]
        )

    # ========================================================
    # SAVE JSON
    # ========================================================

    summary_json_path = (
        output_dir
        / "summary.json"
    )

    with summary_json_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

    # ========================================================
    # MARKDOWN SUMMARY
    # ========================================================

    summary_md_path = (
        output_dir
        / "summary.md"
    )

    comparison = summary[
        "comparison"
    ]

    with summary_md_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "# Static Generative Comparison Pilot\n\n"
        )

        file.write(
            "## Dataset and POD\n\n"
        )

        file.write(
            f"- Parameter configurations: "
            f"{summary['dataset']['n_mu']}\n"
        )

        file.write(
            f"- Realizations per parameter: "
            f"{summary['dataset']['n_realizations']}\n"
        )

        file.write(
            f"- Physical times: "
            f"{summary['dataset']['n_times']}\n"
        )

        file.write(
            f"- Full-order dimension: "
            f"{summary['dataset']['n_dofs']}\n"
        )

        file.write(
            f"- POD modes: "
            f"{summary['pod']['n_modes']}\n"
        )

        file.write(
            f"- POD test trajectory error: "
            f"{100.0 * summary['pod']['test_trajectory_relative_error']:.3f}%\n\n"
        )

        file.write(
            "## Generative comparison\n\n"
        )

        file.write(
            "| Metric | RealNVP | Flow Matching |\n"
        )

        file.write(
            "|---|---:|---:|\n"
        )

        file.write(
            f"| Energy mean | "
            f"{nf_summary['energy_mean']:.6f} | "
            f"{fm_summary['energy_mean']:.6f} |\n"
        )

        file.write(
            f"| Energy median | "
            f"{nf_summary['energy_median']:.6f} | "
            f"{fm_summary['energy_median']:.6f} |\n"
        )

        file.write(
            f"| Energy minimum | "
            f"{nf_summary['energy_min']:.6f} | "
            f"{fm_summary['energy_min']:.6f} |\n"
        )

        file.write(
            f"| Energy maximum | "
            f"{nf_summary['energy_max']:.6f} | "
            f"{fm_summary['energy_max']:.6f} |\n"
        )

        file.write(
            f"| NN relative mean | "
            f"{nf_summary['nearest_neighbor_mean']:.6f} | "
            f"{fm_summary['nearest_neighbor_mean']:.6f} |\n\n"
        )

        file.write(
            "## Main observations\n\n"
        )

        file.write(
            f"- Flow Matching reduces mean energy distance "
            f"by {comparison['fm_energy_mean_reduction_percent']:.2f}% "
            f"relative to RealNVP.\n"
        )

        file.write(
            f"- Flow Matching reduces the mean nearest-neighbor "
            f"relative error by "
            f"{comparison['fm_nn_reduction_percent']:.2f}%.\n"
        )

        if (
            "flow_matching_slowdown"
            in comparison
        ):

            file.write(
                f"- At N={comparison['benchmark_n_samples']}, "
                f"Flow Matching is approximately "
                f"{comparison['flow_matching_slowdown']:.2f}x "
                f"slower than RealNVP.\n"
            )

        file.write(
            "\nThese results refer to the pilot experiment "
            "and should not yet be interpreted as the final "
            "statistical conclusion of the thesis.\n"
        )

    # ========================================================
    # PER-GROUP TABLE
    # ========================================================

    nf_groups = {
        int(row["group_id"]):
            row
        for row in nf_evaluation[
            "per_group"
        ]
    }

    fm_groups = {
        int(row["group_id"]):
            row
        for row in fm_evaluation[
            "per_group"
        ]
    }

    common_groups = sorted(
        set(nf_groups)
        &
        set(fm_groups)
    )

    per_group_rows = []

    for group_id in common_groups:

        nf_row = nf_groups[
            group_id
        ]

        fm_row = fm_groups[
            group_id
        ]

        mu = np.asarray(
            nf_row["mu"]
        )

        per_group_rows.append(
            {
                "group_id":
                    group_id,

                "mu_1":
                    float(mu[0]),

                "mu_2":
                    float(mu[1]),

                "mu_3":
                    float(mu[2]),

                "realnvp_energy":
                    float(
                        nf_row[
                            "energy_distance_squared"
                        ]
                    ),

                "flow_matching_energy":
                    float(
                        fm_row[
                            "energy_distance_squared"
                        ]
                    ),

                "fm_reduction_percent":
                    float(
                        100.0
                        * (
                            nf_row[
                                "energy_distance_squared"
                            ]
                            - fm_row[
                                "energy_distance_squared"
                            ]
                        )
                        /
                        nf_row[
                            "energy_distance_squared"
                        ]
                    ),
            }
        )

    per_group_path = (
        tables_dir
        / "per_group_energy.csv"
    )

    _write_csv(
        per_group_path,
        [
            "group_id",
            "mu_1",
            "mu_2",
            "mu_3",
            "realnvp_energy",
            "flow_matching_energy",
            "fm_reduction_percent",
        ],
        per_group_rows,
    )

    # ========================================================
    # BENCHMARK TABLE
    # ========================================================

    benchmark_rows = []

    for model_name, rows in (
        benchmark.items()
    ):

        for row in rows:

            benchmark_rows.append(
                {
                    "model":
                        model_name,

                    "n_samples":
                        int(
                            row["n_samples"]
                        ),

                    "mean_seconds":
                        float(
                            row["mean_seconds"]
                        ),

                    "median_seconds":
                        float(
                            row[
                                "median_seconds"
                            ]
                        ),

                    "std_seconds":
                        float(
                            row["std_seconds"]
                        ),

                    "seconds_per_sample":
                        float(
                            row[
                                "seconds_per_sample"
                            ]
                        ),

                    "samples_per_second":
                        float(
                            row[
                                "samples_per_second"
                            ]
                        ),
                }
            )

    benchmark_path = (
        tables_dir
        / "benchmark.csv"
    )

    _write_csv(
        benchmark_path,
        [
            "model",
            "n_samples",
            "mean_seconds",
            "median_seconds",
            "std_seconds",
            "seconds_per_sample",
            "samples_per_second",
        ],
        benchmark_rows,
    )

    # ========================================================
    # TIME-WISE TABLE
    # ========================================================

    times = np.asarray(
        times
    )

    timewise_rows = []

    for i, time_value in enumerate(
        times
    ):

        timewise_rows.append(
            {
                "time":
                    float(time_value),

                "realnvp_mean":
                    float(
                        nf_summary[
                            "timewise_mean"
                        ][i]
                    ),

                "flow_matching_mean":
                    float(
                        fm_summary[
                            "timewise_mean"
                        ][i]
                    ),

                "realnvp_median":
                    float(
                        nf_summary[
                            "timewise_median"
                        ][i]
                    ),

                "flow_matching_median":
                    float(
                        fm_summary[
                            "timewise_median"
                        ][i]
                    ),
            }
        )

    timewise_path = (
        tables_dir
        / "timewise_energy.csv"
    )

    _write_csv(
        timewise_path,
        [
            "time",
            "realnvp_mean",
            "flow_matching_mean",
            "realnvp_median",
            "flow_matching_median",
        ],
        timewise_rows,
    )

    # ========================================================
    # TRAINING HISTORY TABLE
    # ========================================================

    training_rows = []

    for model_name, result in [
        ("RealNVP", nf_result),
        ("Flow Matching", fm_result),
    ]:

        train_history = result[
            "train_history"
        ]

        val_history = result[
            "val_history"
        ]

        for epoch, (
            train_loss,
            val_loss,
        ) in enumerate(
            zip(
                train_history,
                val_history,
            ),
            start=1,
        ):

            training_rows.append(
                {
                    "model":
                        model_name,

                    "epoch":
                        epoch,

                    "train_loss":
                        float(train_loss),

                    "val_loss":
                        float(val_loss),
                }
            )

    training_path = (
        tables_dir
        / "training_history.csv"
    )

    _write_csv(
        training_path,
        [
            "model",
            "epoch",
            "train_loss",
            "val_loss",
        ],
        training_rows,
    )

    # ========================================================
    # FIGURE 1 — TIME-WISE ENERGY
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    ax.plot(
        times,
        nf_summary["timewise_mean"],
        label="RealNVP",
    )

    ax.plot(
        times,
        fm_summary["timewise_mean"],
        label="Flow Matching",
    )

    ax.set_xlabel(
        "Physical time"
    )

    ax.set_ylabel(
        "Mean time-wise energy distance squared"
    )

    ax.set_title(
        "Distributional error over time"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    timewise_figure_path = (
        figures_dir
        / "timewise_energy.png"
    )

    fig.savefig(
        timewise_figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # FIGURE 2 — GENERATION TIME
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    for model_name in [
        "RealNVP",
        "Flow Matching",
    ]:

        rows = benchmark[
            model_name
        ]

        sample_sizes = [
            row["n_samples"]
            for row in rows
        ]

        median_times = [
            row["median_seconds"]
            for row in rows
        ]

        ax.plot(
            sample_sizes,
            median_times,
            marker="o",
            label=model_name,
        )

    ax.set_xlabel(
        "Number of generated trajectories"
    )

    ax.set_ylabel(
        "Median generation time [s]"
    )

    ax.set_title(
        "Generative efficiency"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    generation_figure_path = (
        figures_dir
        / "generation_time.png"
    )

    fig.savefig(
        generation_figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # FIGURE 3 — THROUGHPUT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(7, 4)
    )

    for model_name in [
        "RealNVP",
        "Flow Matching",
    ]:

        rows = benchmark[
            model_name
        ]

        sample_sizes = [
            row["n_samples"]
            for row in rows
        ]

        throughput = [
            row[
                "samples_per_second"
            ]
            for row in rows
        ]

        ax.plot(
            sample_sizes,
            throughput,
            marker="o",
            label=model_name,
        )

    ax.set_xlabel(
        "Number of generated trajectories"
    )

    ax.set_ylabel(
        "Trajectories per second"
    )

    ax.set_title(
        "Generation throughput"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    ax.legend()

    fig.tight_layout()

    throughput_figure_path = (
        figures_dir
        / "throughput.png"
    )

    fig.savefig(
        throughput_figure_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # ========================================================
    # FIGURES 4–5 — TRAINING CURVES
    # ========================================================

    training_figure_paths = {}

    for model_name, result, filename in [
        (
            "RealNVP",
            nf_result,
            "training_realnvp.png",
        ),
        (
            "Flow Matching",
            fm_result,
            "training_flow_matching.png",
        ),
    ]:

        fig, ax = plt.subplots(
            figsize=(8, 4)
        )

        ax.plot(
            result[
                "train_history"
            ],
            label="train",
        )

        ax.plot(
            result[
                "val_history"
            ],
            label="validation",
        )

        ax.set_xlabel(
            "Epoch"
        )

        ax.set_ylabel(
            "Training objective"
        )

        ax.set_title(
            f"{model_name} training"
        )

        ax.grid(
            True,
            alpha=0.3,
        )

        ax.legend()

        fig.tight_layout()

        figure_path = (
            figures_dir
            / filename
        )

        fig.savefig(
            figure_path,
            dpi=200,
            bbox_inches="tight",
        )

        plt.close(fig)

        training_figure_paths[
            model_name
        ] = figure_path

    return {
        "summary_json":
            str(
                summary_json_path
            ),

        "summary_markdown":
            str(
                summary_md_path
            ),

        "per_group_table":
            str(
                per_group_path
            ),

        "benchmark_table":
            str(
                benchmark_path
            ),

        "timewise_table":
            str(
                timewise_path
            ),

        "training_table":
            str(
                training_path
            ),

        "timewise_figure":
            str(
                timewise_figure_path
            ),

        "generation_figure":
            str(
                generation_figure_path
            ),

        "throughput_figure":
            str(
                throughput_figure_path
            ),

        "training_figures":
            {
                key:
                    str(value)
                for key, value
                in training_figure_paths.items()
            },

        "summary":
            summary,
    }