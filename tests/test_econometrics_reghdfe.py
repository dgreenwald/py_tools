import pandas as pd
import pytest

from py_tools.econometrics.reghdfe import reghdfe_formula


def test_reghdfe_formula_cluster_not_implemented():
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 0.0], "g": [1, 1, 2]})
    with pytest.raises(NotImplementedError, match="cluster-robust"):
        reghdfe_formula(df, "y ~ x", cluster="g")
