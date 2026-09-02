import time
import numpy as np
import torch


def _synchronize_surrogate(surrogate):
    """
    Synchronize CUDA operations before/after timing.

    This is necessary because GPU operations are asynchronous.
    """

    try:
        device = next(
            surrogate.model.parameters()
        ).device
    except StopIteration:
        return

    if (
        device.type == "cuda"
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize(
            device
        )


def benchmark_generation(
    surrogate,
    mu,
    times,
    sample_sizes=(1, 10, 100),
    n_repeats=5,
    n_warmup=2,
    temporal_coupling="shared_noise",
):
    """
    Benchmark the complete generation of physical trajectories.

    The measured operation is

        (mu, times)
            ->
        conditional generative model
            ->
        POD coefficient denormalization
            ->
        POD reconstruction
            ->
        physical trajectories

    Therefore, this measures the actual online cost faced by
    a user of the surrogate.

    Parameters
    ----------
    surrogate
        StaticPODGenerativeSurrogate.

    mu
        Conditioning parameter vector.

    times
        Physical time grid.

    sample_sizes
        Number of stochastic trajectories to generate
        in each benchmark.

    n_repeats
        Number of repeated measurements for every sample size.

    n_warmup
        Number of preliminary generations excluded from timing.

    temporal_coupling
        "shared_noise" or "independent".

    Returns
    -------
    list of dict
        Timing statistics for every sample size.
    """

    sample_sizes = [
        int(n)
        for n in sample_sizes
    ]

    if any(
        n <= 0
        for n in sample_sizes
    ):
        raise ValueError(
            "All sample_sizes must be positive."
        )

    if n_repeats <= 0:
        raise ValueError(
            "n_repeats must be positive."
        )

    if n_warmup < 0:
        raise ValueError(
            "n_warmup cannot be negative."
        )

    # --------------------------------------------------------
    # WARM-UP
    #
    # Especially important on GPU:
    # exclude initialization overhead from the benchmark.
    # --------------------------------------------------------

    for _ in range(n_warmup):

        _ = surrogate.sample(
            mu=mu,
            times=times,
            n_samples=1,
            temporal_coupling=temporal_coupling,
        )

    _synchronize_surrogate(
        surrogate
    )

    results = []

    # --------------------------------------------------------
    # BENCHMARK EACH REQUESTED SAMPLE SIZE
    # --------------------------------------------------------

    for n_samples in sample_sizes:

        durations = []

        for _ in range(n_repeats):

            _synchronize_surrogate(
                surrogate
            )

            start = time.perf_counter()

            samples = surrogate.sample(
                mu=mu,
                times=times,
                n_samples=n_samples,
                temporal_coupling=temporal_coupling,
            )

            _synchronize_surrogate(
                surrogate
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            # Basic sanity checks.
            if samples.shape[0] != n_samples:
                raise RuntimeError(
                    "The surrogate returned an unexpected "
                    "number of samples."
                )

            if not np.isfinite(
                samples
            ).all():
                raise RuntimeError(
                    "The surrogate produced non-finite samples."
                )

            durations.append(
                elapsed
            )

            # We do not need to keep generated trajectories.
            del samples

        durations = np.asarray(
            durations,
            dtype=np.float64,
        )

        mean_seconds = float(
            durations.mean()
        )

        median_seconds = float(
            np.median(
                durations
            )
        )

        std_seconds = float(
            durations.std()
        )

        seconds_per_sample = (
            median_seconds
            / n_samples
        )

        samples_per_second = (
            n_samples
            / median_seconds
        )

        results.append(
            {
                "n_samples":
                    int(n_samples),

                "mean_seconds":
                    mean_seconds,

                "median_seconds":
                    median_seconds,

                "std_seconds":
                    std_seconds,

                "seconds_per_sample":
                    float(
                        seconds_per_sample
                    ),

                "samples_per_second":
                    float(
                        samples_per_second
                    ),

                "all_times":
                    durations,
            }
        )

    return results


def benchmark_surrogates(
    surrogates,
    mu,
    times,
    sample_sizes=(1, 10, 100),
    n_repeats=5,
    n_warmup=2,
    temporal_coupling="shared_noise",
):
    """
    Benchmark several surrogates using exactly the same
    experimental settings.

    Parameters
    ----------
    surrogates
        Dictionary such as

            {
                "NF": nf_surrogate,
                "FM": fm_surrogate,
            }

    Returns
    -------
    dict
        Benchmark results indexed by model name.
    """

    results = {}

    for name, surrogate in (
        surrogates.items()
    ):

        print(
            f"\nBenchmarking {name}..."
        )

        model_results = (
            benchmark_generation(
                surrogate=surrogate,
                mu=mu,
                times=times,
                sample_sizes=sample_sizes,
                n_repeats=n_repeats,
                n_warmup=n_warmup,
                temporal_coupling=
                    temporal_coupling,
            )
        )

        results[name] = (
            model_results
        )

        for result in model_results:

            print(
                f"  N="
                f"{result['n_samples']:5d} | "
                f"median="
                f"{result['median_seconds']:.6f} s | "
                f"per sample="
                f"{result['seconds_per_sample']:.6e} s | "
                f"throughput="
                f"{result['samples_per_second']:.2f} samples/s"
            )

    return results