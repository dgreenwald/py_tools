import numpy as np
import pandas as pd

from py_tools.datasets.fama_bliss import load


def test_load_selects_discount_bonds_and_accepts_filename(tmp_path):
    rows = []
    for date in ("2020/01/31", "2020/02/28"):
        for term in range(5001, 5006):
            rows.append(
                {"TIDXFAM": "DISCBOND", "TTERMTYPE": term, "MCALDT": date,
                 "TMNOMPRC": 100.0 * np.exp(-0.01 * (term - 5000))}
            )
    rows.append({"TIDXFAM": "RISKFREE", "TTERMTYPE": 30, "MCALDT": None,
                 "TMNOMPRC": None})
    pd.DataFrame(rows).to_csv(tmp_path / "fama_bliss.csv", index=False)

    result = load(tmp_path, filename="fama_bliss.csv")
    assert result.shape == (2, 15)
    assert set(result.columns) == {
        *(f"P{i}" for i in range(1, 6)),
        *(f"p{i}" for i in range(1, 6)),
        *(f"y{i}" for i in range(1, 6)),
    }
    np.testing.assert_allclose(result[[f"y{i}" for i in range(1, 6)]], 0.01)

    (tmp_path / "fama_bliss.csv").rename(tmp_path / "fbdata.csv")
    pd.testing.assert_frame_equal(load(tmp_path), result)
