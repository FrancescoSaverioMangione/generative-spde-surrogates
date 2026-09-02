from pathlib import Path

import yaml


def load_config(path):
    """
    Load an experiment configuration from a YAML file.

    Parameters
    ----------
    path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed experiment configuration.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        config = yaml.safe_load(
            file
        )

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {path}"
        )

    return config