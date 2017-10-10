import numpy as np

def gradient(f, x, args=(), step=1e-5):

    grad = None
    for ii in range(len(x)):

        x[ii] += step
        f_hi = f(x, *args)
        x[ii] -= (2.0 * step)
        f_lo = f(x, *args)
        x[ii] += step

        df_i = np.array(f_hi - f_lo) / (2.0 * step)

        if grad is None:

            if df_i.shape == ():
                ncols = 1 
            else:
                ncols = len(df_i)

            grad = np.zeros((len(x), ncols))

        grad[ii, :] = df_i

    return grad

