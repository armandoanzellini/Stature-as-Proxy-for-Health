import numpy as np
import pandas as pd
import pytest
import StatureHealthBayes as shb

@pytest.fixture
def toy():
    return pd.DataFrame(
        {"N": [20, 25, 18, 30, 22],
         "Mean": [160, 165, 158, 170, 162],
         "STD": [4.0, 4.5, 3.8, 5.0, 4.2]},
        index=["A", "B", "C", "D", "E"],
    )

def test_quartiles_are_symmetric_about_mean():
    assert shb.q1(160, 4) == pytest.approx(160 - 0.6745 * 4)
    assert shb.q3(160, 4) == pytest.approx(160 + 0.6745 * 4)

def test_anova_runs_and_writes_outputs(toy, tmp_path):
    shb.bayes_ANOVA(toy, "Test", out_dir=tmp_path, savefig=True,
                    draws=200, tune=200, chains=2)
    assert (tmp_path / "ANOVAstats_Test.xlsx").exists()
    assert (tmp_path / "PairwiseComp_Test.xlsx").exists()

def test_best_runs(toy, tmp_path):
    shb.BEST(toy.iloc[0], toy.iloc[1], "Test", out_dir=tmp_path,
             savefig=True, draws=200, tune=200, chains=2)
    assert (tmp_path / "BEST_Test.png").exists()