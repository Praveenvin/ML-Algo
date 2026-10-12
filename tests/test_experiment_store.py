"""Tests for the app-facing experiment persistence layer. Uses monkeypatch to
redirect writes to a temporary directory so tests never touch the project's
real results/ directory."""

from pathlib import Path

import pandas as pd
import pytest

from src import experiment_store


@pytest.fixture
def isolated_metrics_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(experiment_store.config, "METRICS_DIR", tmp_path)
    monkeypatch.setattr(experiment_store.config, "PROJECT_ROOT", tmp_path.parent)
    monkeypatch.setattr(experiment_store, "EXPERIMENT_INDEX_PATH", tmp_path / "experiment_index.csv")
    return tmp_path


def _sample_metrics_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"model": ["Logistic Regression", "KNN"], "accuracy": [0.8, 0.85], "f1_score": [0.7, 0.82]}
    )


def test_save_experiment_writes_metrics_and_index(isolated_metrics_dir: Path) -> None:
    experiment_id = experiment_store.save_experiment(
        task="classification", dataset_name="demo.csv", metrics_df=_sample_metrics_df(),
        best_model="KNN", selection_metric="cv f1", target_column="Purchased",
    )
    assert experiment_id
    assert experiment_store.EXPERIMENT_INDEX_PATH.exists()

    index_df = experiment_store.load_experiment_index()
    assert len(index_df) == 1
    assert index_df.iloc[0]["experiment_id"] == experiment_id
    assert index_df.iloc[0]["best_model"] == "KNN"


def test_save_experiment_appends_without_overwriting(isolated_metrics_dir: Path) -> None:
    experiment_store.save_experiment("classification", "a.csv", _sample_metrics_df())
    experiment_store.save_experiment("regression", "b.csv", _sample_metrics_df())

    index_df = experiment_store.load_experiment_index()
    assert len(index_df) == 2
    assert set(index_df["task"]) == {"classification", "regression"}


def test_load_experiment_metrics_round_trips(isolated_metrics_dir: Path) -> None:
    original = _sample_metrics_df()
    experiment_id = experiment_store.save_experiment("classification", "a.csv", original)

    loaded = experiment_store.load_experiment_metrics(experiment_id)
    pd.testing.assert_frame_equal(loaded, original)


def test_load_experiment_metrics_raises_for_unknown_id(isolated_metrics_dir: Path) -> None:
    with pytest.raises(ValueError, match="No experiment found"):
        experiment_store.load_experiment_metrics("does-not-exist")


def test_load_all_results_filters_by_task(isolated_metrics_dir: Path) -> None:
    experiment_store.save_experiment("classification", "a.csv", _sample_metrics_df())
    experiment_store.save_experiment("regression", "b.csv", _sample_metrics_df())

    classification_only = experiment_store.load_all_results(task="classification")
    assert (classification_only["task"] == "classification").all()
    assert len(classification_only) == 2  # 2 models in the sample metrics df

    everything = experiment_store.load_all_results()
    assert len(everything) == 4


def test_load_all_results_empty_when_no_experiments(isolated_metrics_dir: Path) -> None:
    result = experiment_store.load_all_results()
    assert result.empty


def test_load_experiment_index_empty_when_missing(isolated_metrics_dir: Path) -> None:
    index_df = experiment_store.load_experiment_index()
    assert index_df.empty
    assert "experiment_id" in index_df.columns
