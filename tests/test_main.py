import logging

import pandas as pd

from cointegration_analysis.cli import load_pair_data


def test_load_pair_data_sorts_and_deduplicates(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    csv_path = tmp_path / "demo_pair_data.csv"
    df = pd.DataFrame(
        {"A": [1.0, 2.0, 3.0], "B": [2.0, 3.0, 4.0]},
        index=pd.to_datetime(["2024-01-03", "2024-01-01", "2024-01-01"]),
    )
    df.to_csv(csv_path)

    data = load_pair_data(["demo_pair"], data_dir=tmp_path)
    loaded = data["demo_pair"]

    assert loaded.index.is_monotonic_increasing
    assert loaded.index.is_unique
    assert "Dropping duplicated timestamps" in caplog.text
