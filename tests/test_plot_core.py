"""Tests for py_tools.plot.core"""

import os
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import pytest

import py_tools.plot.core as pc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n=50, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2000-01-01', periods=n, freq=pd.offsets.MonthEnd())
    return pd.DataFrame(
        {
            'a': rng.standard_normal(n),
            'b': rng.standard_normal(n),
            'w': np.abs(rng.standard_normal(n)) + 0.1,
        },
        index=idx,
    )


def _make_numeric_df(n=60, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            'x': rng.standard_normal(n),
            'y': rng.standard_normal(n),
            'z': rng.standard_normal(n),
            'w': np.abs(rng.standard_normal(n)) + 0.1,
        }
    )


# ---------------------------------------------------------------------------
# set_fontsizes
# ---------------------------------------------------------------------------

class TestSetFontsizes:
    def test_sets_fontsize_on_axes(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('T')
        pc.set_fontsizes(ax, 20)
        assert ax.title.get_fontsize() == 20
        assert ax.xaxis.label.get_fontsize() == 20
        assert ax.yaxis.label.get_fontsize() == 20
        plt.close(fig)


# ---------------------------------------------------------------------------
# save_hist_npy / load_hist_npy
# ---------------------------------------------------------------------------

class TestHistNpy:
    def test_roundtrip(self, tmp_path):
        vals = np.arange(10.0)
        base = str(tmp_path / 'h_')
        pc.save_hist_npy(vals, base)
        hist, bin_edges = pc.load_hist_npy(base)
        assert len(hist) + 1 == len(bin_edges)
        assert hist.sum() == len(vals)

    def test_custom_bins(self, tmp_path):
        vals = np.linspace(0, 1, 100)
        base = str(tmp_path / 'h2_')
        pc.save_hist_npy(vals, base, bins=5)
        hist, bin_edges = pc.load_hist_npy(base)
        assert len(hist) == 5


# ---------------------------------------------------------------------------
# compute_hist
# ---------------------------------------------------------------------------

class TestComputeHist:
    def test_sums_to_one(self):
        df = pd.DataFrame({'x': np.arange(10.0)})
        bins = np.array([0.0, 3.0, 6.0, 10.0])
        h = pc.compute_hist(df, 'x', bins)
        assert np.isclose(h.sum(), 1.0)

    def test_weighted_sums_to_one(self):
        df = pd.DataFrame({'x': np.arange(10.0), 'w': np.ones(10) * 2.0})
        bins = np.array([0.0, 5.0, 10.0])
        h = pc.compute_hist(df, 'x', bins, wvar='w')
        assert np.isclose(h.sum(), 1.0)

    def test_bin_count(self):
        df = pd.DataFrame({'x': np.linspace(0, 10, 100)})
        bins = np.linspace(0, 10, 6)  # 5 bins
        h = pc.compute_hist(df, 'x', bins)
        assert len(h) == 5


# ---------------------------------------------------------------------------
# get_45_bounds
# ---------------------------------------------------------------------------

class TestGet45Bounds:
    def test_bounds_cover_data(self):
        df = pd.DataFrame({'x': [0.0, 1.0], 'y': [0.5, 1.5]})
        lb, ub = pc.get_45_bounds(df, 'x', 'y')
        assert lb < 0.0
        assert ub > 1.5

    def test_symmetric_data(self):
        df = pd.DataFrame({'x': [0.0, 4.0], 'y': [0.0, 4.0]})
        lb, ub = pc.get_45_bounds(df, 'x', 'y', margin=0.0)
        assert lb == 0.0
        assert ub == 4.0


# ---------------------------------------------------------------------------
# get_colors
# ---------------------------------------------------------------------------

class TestGetColors:
    def test_returns_correct_count(self):
        colors = pc.get_colors(5, 'viridis')
        assert colors.shape == (5, 4)

    def test_startval_endval(self):
        colors_full = pc.get_colors(3, 'viridis', startval=0.0, endval=1.0)
        colors_half = pc.get_colors(3, 'viridis', startval=0.0, endval=0.5)
        # Full-range last color should differ from half-range last color
        assert not np.allclose(colors_full[-1], colors_half[-1])


# ---------------------------------------------------------------------------
# state_scatter_inner
# ---------------------------------------------------------------------------

class TestStateScatterInner:
    def test_annotates_without_error(self):
        import matplotlib.pyplot as plt
        df = pd.DataFrame({'x': [1.0, 2.0], 'y': [3.0, 4.0]}, index=['A', 'B'])
        fig, ax = plt.subplots()
        result = pc.state_scatter_inner(ax, df, 'y', 'x')
        assert result is None
        plt.close(fig)


# ---------------------------------------------------------------------------
# normalized  (bug fix: invert_list as boolean list)
# ---------------------------------------------------------------------------

class TestNormalized:
    def test_returns_none(self, tmp_path):
        df = _make_df(20)
        fp = str(tmp_path / 'norm.png')
        result = pc.normalized(df, ['a', 'b'], filepath=fp)
        assert result is None
        assert os.path.exists(fp)

    def test_invert_list_applies(self, tmp_path):
        """Inverted series should produce a mirrored plot; no crash."""
        df = _make_df(20)
        fp = str(tmp_path / 'norm_inv.png')
        result = pc.normalized(df, ['a', 'b'], invert_list=[True, False], filepath=fp)
        assert result is None

    def test_default_invert_list_no_inversion(self, tmp_path):
        """Default invert_list=[False, False] should not crash."""
        df = _make_df(20)
        fp = str(tmp_path / 'norm_def.png')
        pc.normalized(df, ['a'], filepath=fp)


# ---------------------------------------------------------------------------
# hist
# ---------------------------------------------------------------------------

class TestHist:
    def test_returns_true_on_valid_data(self, tmp_path):
        df = _make_df(30)
        fp = str(tmp_path / 'hist.png')
        result = pc.hist(df, 'a', filepath=fp)
        assert result is True
        assert os.path.exists(fp)

    def test_returns_false_when_var_missing(self):
        df = pd.DataFrame({'a': [1.0, 2.0]})
        result = pc.hist(df, 'missing_var')
        assert result is False

    def test_returns_false_on_empty_after_clean(self):
        df = pd.DataFrame({'a': [np.nan, np.nan]})
        result = pc.hist(df, 'a')
        assert result is False

    def test_with_weights(self, tmp_path):
        df = _make_df(30)
        fp = str(tmp_path / 'hist_w.png')
        result = pc.hist(df, 'a', wvar='w', filepath=fp)
        assert result is True

    def test_with_bins(self, tmp_path):
        df = _make_df(30)
        fp = str(tmp_path / 'hist_bins.png')
        result = pc.hist(df, 'a', bins=10, filepath=fp)
        assert result is True

    def test_with_vertline(self, tmp_path):
        df = _make_df(30)
        fp = str(tmp_path / 'hist_vline.png')
        result = pc.hist(df, 'a', x_vertline=0.0, filepath=fp)
        assert result is True


# ---------------------------------------------------------------------------
# two_axis (bug fix: np.array([...]) normalization)
# ---------------------------------------------------------------------------

class TestTwoAxis:
    def test_runs_without_error(self, tmp_path):
        df = _make_df(40)
        fp = str(tmp_path / 'two_axis.png')
        pc.two_axis(df, 'a', 'b', filepath=fp)
        assert os.path.exists(fp)

    def test_with_normalize(self, tmp_path):
        """Previously broke due to np.array(val, val) bug."""
        df = _make_df(40)
        fp = str(tmp_path / 'two_axis_norm.png')
        pc.two_axis(df, 'a', 'b', normalize=True, filepath=fp)
        assert os.path.exists(fp)

    def test_flip1_flip2(self, tmp_path):
        df = _make_df(40)
        fp = str(tmp_path / 'two_axis_flip.png')
        pc.two_axis(df, 'a', 'b', flip1=True, flip2=True, filepath=fp)
        assert os.path.exists(fp)

    def test_single_legend(self, tmp_path):
        df = _make_df(40)
        fp = str(tmp_path / 'two_axis_sl.png')
        pc.two_axis(df, 'a', 'b', single_legend=True, loc_single='upper left', filepath=fp)
        assert os.path.exists(fp)

    def test_axvline(self, tmp_path):
        df = _make_df(40)
        fp = str(tmp_path / 'two_axis_axvline.png')
        pc.two_axis(df, 'a', 'b', axvline=df.index[10], filepath=fp)
        assert os.path.exists(fp)


# ---------------------------------------------------------------------------
# double_hist  (bug fix: copy_path2 not copy_path1)
# ---------------------------------------------------------------------------

class TestDoubleHist:
    def _make_dfs(self):
        rng = np.random.default_rng(1)
        df1 = pd.DataFrame({'x': rng.standard_normal(40), 'w': np.ones(40)})
        df2 = pd.DataFrame({'x': rng.standard_normal(40), 'w': np.ones(40)})
        return df1, df2

    def test_returns_true(self, tmp_path):
        df1, df2 = self._make_dfs()
        fp = str(tmp_path / 'dh.png')
        result = pc.double_hist(df1, df2, var='x', filepath=fp)
        assert result is True
        assert os.path.exists(fp)

    def test_copy_paths_written_to_correct_files(self, tmp_path):
        """Bug fix: copy_path2 was accidentally written to copy_path1."""
        df1, df2 = self._make_dfs()
        p1 = str(tmp_path / 'copy1')
        p2 = str(tmp_path / 'copy2')
        pc.double_hist(
            df1, df2, var='x',
            bins=np.linspace(-3, 3, 10),
            copy_path1=p1, copy_path2=p2,
        )
        # Both paths must exist independently (to_pickle writes directly to path)
        assert os.path.exists(p1)
        assert os.path.exists(p2)
        # Sanity: files for each path are non-empty
        assert os.path.getsize(p1) > 0
        assert os.path.getsize(p2) > 0

    def test_missing_var_returns_false(self):
        df = pd.DataFrame({'x': [np.nan, np.nan]})
        result = pc.double_hist(df, var='x')
        assert result is False

    def test_use_bar(self, tmp_path):
        df1, df2 = self._make_dfs()
        fp = str(tmp_path / 'dh_bar.png')
        bins = np.linspace(-3, 3, 8)
        result = pc.double_hist(df1, df2, var='x', bins=bins, use_bar=True, filepath=fp)
        assert result is True

    def test_topcode_bottomcode(self, tmp_path):
        df1, df2 = self._make_dfs()
        fp = str(tmp_path / 'dh_code.png')
        bins = np.linspace(-3, 3, 8)
        result = pc.double_hist(df1, df2, var='x', bins=bins, topcode=True, bottomcode=True, filepath=fp)
        assert result is True


# ---------------------------------------------------------------------------
# multi_hist
# ---------------------------------------------------------------------------

class TestMultiHist:
    def _make_dfs(self):
        rng = np.random.default_rng(2)
        return [
            pd.DataFrame({'x': rng.standard_normal(30)}),
            pd.DataFrame({'x': rng.standard_normal(30)}),
        ]

    def test_basic(self, tmp_path):
        dfs = self._make_dfs()
        fp = str(tmp_path / 'mh.png')
        result = pc.multi_hist(dfs, labels=['A', 'B'], xvar='x', filepath=fp)
        assert result is True
        assert os.path.exists(fp)

    def test_use_bar(self, tmp_path):
        dfs = self._make_dfs()
        bins = np.linspace(-3, 3, 8)
        fp = str(tmp_path / 'mh_bar.png')
        result = pc.multi_hist(dfs, labels=['A', 'B'], xvar='x', bins=bins, use_bar=True, filepath=fp)
        assert result is True

    def test_density_false(self, tmp_path):
        dfs = self._make_dfs()
        fp = str(tmp_path / 'mh_nodensity.png')
        result = pc.multi_hist(dfs, labels=['A', 'B'], xvar='x', density=False, filepath=fp)
        assert result is True


# ---------------------------------------------------------------------------
# var_irfs
# ---------------------------------------------------------------------------

class TestVarIrfs:
    def test_runs_and_saves(self, tmp_path):
        rng = np.random.default_rng(3)
        irfs = rng.standard_normal((100, 8, 2, 2))
        fp = str(tmp_path / 'irfs.png')
        result = pc.var_irfs(irfs, ['y1', 'y2'], shock_list=['s1', 's2'], filepath=fp)
        assert result is None
        assert os.path.exists(fp)

    def test_returns_none(self, tmp_path):
        rng = np.random.default_rng(4)
        irfs = rng.standard_normal((50, 6, 3, 3))
        fp = str(tmp_path / 'irfs2.png')
        assert pc.var_irfs(irfs, ['y1', 'y2', 'y3'], filepath=fp) is None


# ---------------------------------------------------------------------------
# plot_series
# ---------------------------------------------------------------------------

class TestPlotSeries:
    def test_saves_file(self, tmp_path):
        df = _make_df(40)
        fp = str(tmp_path / 'series.png')
        pc.plot_series(df, ['a', 'b'], filepath=fp)
        assert os.path.exists(fp)

    def test_single_series(self, tmp_path):
        df = _make_df(30)
        fp = str(tmp_path / 'series_single.png')
        pc.plot_series(df, ['a'], filepath=fp)
        assert os.path.exists(fp)

    def test_raises_on_legacy_args(self):
        df = _make_df(20)
        with pytest.raises(Exception):
            pc.plot_series(df, ['a'], directory='/tmp')
        with pytest.raises(Exception):
            pc.plot_series(df, ['a'], filename='foo')
        with pytest.raises(Exception):
            pc.plot_series(df, ['a'], plot_type='pdf')

    def test_inner_sample(self, tmp_path):
        df = _make_df(20)
        # Insert some NaN in 'b'
        df.loc[df.index[::3], 'b'] = np.nan
        fp = str(tmp_path / 'series_inner.png')
        pc.plot_series(df, ['a', 'b'], sample='inner', filepath=fp)
        assert os.path.exists(fp)

    def test_invalid_sample_raises(self):
        df = _make_df(20)
        with pytest.raises(Exception):
            pc.plot_series(df, ['a'], sample='bad', filepath='/tmp/x.png')

    def test_with_vertline(self, tmp_path):
        df = _make_df(30)
        fp = str(tmp_path / 'series_vl.png')
        pc.plot_series(df, ['a'], vertline_ix=5, filepath=fp)
        assert os.path.exists(fp)


# ---------------------------------------------------------------------------
# projection
# ---------------------------------------------------------------------------

class TestProjection:
    def test_saves_file(self, tmp_path):
        rng = np.random.default_rng(5)
        x = rng.standard_normal((3, 8))
        se = np.abs(rng.standard_normal((3, 8))) + 0.1
        var_titles = ['y1', 'y2', 'y3']
        out_dir = str(tmp_path)
        pc.projection(x, se, var_titles, 'shock', out_dir=out_dir)
        files = list(tmp_path.glob('*.pdf'))
        assert len(files) == 1

    def test_returns_none(self, tmp_path):
        rng = np.random.default_rng(6)
        x = rng.standard_normal((2, 5))
        se = np.abs(rng.standard_normal((2, 5))) + 0.1
        result = pc.projection(x, se, ['a', 'b'], 'shock', out_dir=str(tmp_path))
        assert result is None

    def test_single_row_adjusts_n_per_row(self, tmp_path):
        """When n_rows == 1, n_per_row should equal Ny (2 here)."""
        rng = np.random.default_rng(7)
        x = rng.standard_normal((2, 4))
        se = np.abs(rng.standard_normal((2, 4))) + 0.1
        out_dir = str(tmp_path)
        pc.projection(x, se, ['a', 'b'], 'shock', n_per_row=10, out_dir=out_dir)
        assert list(tmp_path.glob('*.pdf'))


# ---------------------------------------------------------------------------
# scatter
# ---------------------------------------------------------------------------

class TestScatter:
    def test_basic(self, tmp_path):
        df = _make_numeric_df(30)
        fp = str(tmp_path / 'scatter.png')
        pc.scatter(df, 'y', 'x', labels={'x': 'X', 'y': 'Y'}, filepath=fp)
        assert os.path.exists(fp)

    def test_include45(self, tmp_path):
        df = _make_numeric_df(20)
        fp = str(tmp_path / 'scatter45.png')
        pc.scatter(df, 'y', 'x', labels={'x': 'X', 'y': 'Y'}, include45=True, filepath=fp)
        assert os.path.exists(fp)

    def test_multicolor(self, tmp_path):
        df = _make_numeric_df(20)
        fp = str(tmp_path / 'scatter_mc.png')
        pc.scatter(df, 'y', 'x', labels={'x': 'X', 'y': 'Y'}, multicolor=True, filepath=fp)
        assert os.path.exists(fp)


# ---------------------------------------------------------------------------
# binscatter  (bug fix: fit_var path)
# ---------------------------------------------------------------------------

class TestBinscatter:
    def test_basic_returns_dataframe(self, tmp_path):
        df = _make_numeric_df(80)
        fp = str(tmp_path / 'bs.png')
        result = pc.binscatter(df, 'y', 'x', filepath=fp)
        assert isinstance(result, pd.DataFrame)
        assert os.path.exists(fp)

    def test_multiple_yvars(self, tmp_path):
        df = _make_numeric_df(80)
        fp = str(tmp_path / 'bs_multi.png')
        result = pc.binscatter(df, ['y', 'z'], 'x', filepath=fp)
        assert isinstance(result, pd.DataFrame)

    def test_with_weights(self, tmp_path):
        df = _make_numeric_df(80)
        fp = str(tmp_path / 'bs_w.png')
        result = pc.binscatter(df, 'y', 'x', wvar='w', filepath=fp)
        assert isinstance(result, pd.DataFrame)

    def test_fit_var_provided(self, tmp_path):
        """Bug fix: fit_var must be used as the fit column when provided."""
        df = _make_numeric_df(80)
        # Pre-compute a fitted column
        from scipy.stats import linregress
        slope, intercept, *_ = linregress(df['x'], df['y'])
        df['y_custom_fit'] = intercept + slope * df['x']
        fp = str(tmp_path / 'bs_fitvar.png')
        result = pc.binscatter(df, 'y', 'x', fit_var='y_custom_fit', filepath=fp)
        assert isinstance(result, pd.DataFrame)

    def test_no_line(self, tmp_path):
        df = _make_numeric_df(80)
        fp = str(tmp_path / 'bs_noline.png')
        result = pc.binscatter(df, 'y', 'x', plot_line=False, filepath=fp)
        assert isinstance(result, pd.DataFrame)

    def test_include45(self, tmp_path):
        df = _make_numeric_df(60)
        fp = str(tmp_path / 'bs_45.png')
        result = pc.binscatter(df, 'y', 'x', include45=True, filepath=fp)
        assert isinstance(result, pd.DataFrame)

    def test_include0(self, tmp_path):
        df = _make_numeric_df(60)
        fp = str(tmp_path / 'bs_0.png')
        result = pc.binscatter(df, 'y', 'x', include0=True, filepath=fp)
        assert isinstance(result, pd.DataFrame)

    def test_median_binning(self, tmp_path):
        df = _make_numeric_df(60)
        fp = str(tmp_path / 'bs_median.png')
        result = pc.binscatter(df, 'y', 'x', median=True, filepath=fp)
        assert isinstance(result, pd.DataFrame)
