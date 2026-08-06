"""Tests for py_tools.econ.aim"""

import numpy as np

from py_tools.econ.aim import AimObj


class TestAimObjAr1:
    """Test AimObj on a simple AR(1) model: x_t = rho * x_{t-1} + eps_t.

    AIM format H * [x_{t-1}, x_t, x_{t+1}]' = 0, with nlead=1.
    For the AR(1) case: -rho * x_{t-1} + x_t + 0 * x_{t+1} = eps_t
    => H = [[-rho, 1, 0]]
    Expected reduced form: B = [[rho]]
    """

    def setup_method(self):
        self.rho = 0.8
        self.H = np.array([[-self.rho, 1.0, 0.0]])

    def test_solve_runs(self):
        aim = AimObj(self.H.copy(), nlead=1)
        aim.solve()

    def test_B_shape(self):
        aim = AimObj(self.H.copy(), nlead=1)
        aim.solve()
        # B should be (neq, neq * nlag) = (1, 1)
        assert aim.B.shape == (1, 1)

    def test_B_recovers_ar_coefficient(self):
        aim = AimObj(self.H.copy(), nlead=1)
        aim.solve()
        assert np.isclose(aim.B[0, 0], self.rho, atol=1e-8)

    def test_B_shape_high_persistence(self):
        H = np.array([[-0.95, 1.0, 0.0]])
        aim = AimObj(H.copy(), nlead=1)
        aim.solve()
        assert aim.B.shape == (1, 1)
        assert np.isclose(aim.B[0, 0], 0.95, atol=1e-8)


class TestAimObjAttributes:
    def test_neq_set_correctly(self):
        H = np.array([[-0.5, 1.0, 0.0]])
        aim = AimObj(H.copy(), nlead=1)
        assert aim.neq == 1

    def test_nlag_set_correctly(self):
        H = np.array([[-0.5, 1.0, 0.0]])
        aim = AimObj(H.copy(), nlead=1)
        assert aim.nlag == 1

    def test_B_is_set_after_solve(self):
        H = np.array([[-0.7, 1.0, 0.0]])
        aim = AimObj(H.copy(), nlead=1)
        aim.solve()
        assert hasattr(aim, "B")
        assert aim.B is not None
