import scikit.learn.linear_model as lm
import py_tools.time_series as ts

def lasso(df_in, lhs, rhs, match='inner', ix=None, **kwargs):

    df = df_in[[[lhs] + rhs]].copy()

    if 'const' in rhs and 'const' not in df:
        df['const'] = 1.0

    X = df.ix[:, rhs].values
    z = df.ix[:, lhs].values

    ix, Xs, zs = ts.match_xy(X, z, how=match, ix=ix)



    print(results.summary())

    return None
