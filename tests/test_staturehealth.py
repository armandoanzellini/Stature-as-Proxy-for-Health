"""Tests for StatureHealthBayes, run on small synthetic data (the real data is private)."""
import matplotlib.pyplot as plt
import pandas as pd
import pymc as pm
import pytest

import StatureHealthBayes as shb


@pytest.fixture
def toy():
    return pd.DataFrame(
        {"N":    [20, 25, 18, 30, 22],
         "Mean": [160.0, 165.0, 158.0, 170.0, 162.0],
         "STD":  [4.0, 4.5, 3.8, 5.0, 4.2],
         "Min":  [150.0, 154.0, 149.0, 158.0, 152.0],
         "Max":  [170.0, 176.0, 167.0, 181.0, 172.0]},
        index=pd.Index(["A", "B", "C", "D", "E"], name="Site"),
    )


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    """Shrink sampling and figure resolution for every test."""
    sample, savefig = pm.sample, plt.savefig
    monkeypatch.setattr(pm, "sample", lambda *a, **k: sample(
        **{**k, "draws": 200, "tune": 200, "chains": 2, "cores": 1, "progressbar": False}))
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: savefig(*a, **{**k, "dpi": 50}))


def test_boxplots(toy, tmp_path):
    shb.plot_boxplots(toy, toy, ["A", "B", "C"], out_dir=tmp_path)
    assert len(list(tmp_path.iterdir())) == 1


def test_bayes_anova(toy, tmp_path):
    shb.bayes_ANOVA(toy, "Test", out_dir=tmp_path)
    assert len(list(tmp_path.iterdir())) == 4  # stats table, pairwise table, figure

    summary = pd.read_excel(tmp_path / "ANOVAstats_Test.xlsx", index_col=0)
    assert summary.loc["A", "mean"] == pytest.approx(160.0, abs=2.0)  # site labels line up


def test_best(toy, tmp_path):
    shb.BEST(toy.loc["A"], toy.loc["B"], "Test", out_dir=tmp_path)
    assert len(list(tmp_path.iterdir())) == 4  # stats table, two figures

    summary = pd.read_excel(tmp_path / "BESTstats_Test.xlsx", index_col=0)
    assert summary.loc["diff_means", "mean"] == pytest.approx(-5.0, abs=1.5)  # 160 - 165