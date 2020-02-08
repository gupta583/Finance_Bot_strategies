from math import sqrt, log
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy
import pywt
import pandas as pd
import sklearn
import sklearn.metrics
import os

def brownian(x0, n, dt, delta, out=None):
    """\
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = numpy.asarray(x0);

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta*sqrt(dt));

    # If `out` was not given, create an output array.
    if out is None:
        out = numpy.empty(r.shape);

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    numpy.cumsum(r, axis=-1, out=out);

    # Add the initial condition.
    out += numpy.expand_dims(x0, axis=-1);

    return out;

def getBM():
    # The Wiener process parameter.
    delta = 2;
    # Number of steps.
    N = 512;
    N = 1000000;
    # Total time.
    T = 10.0;
    T = N;
    # Time step size
    dt = T/N;
    # Create an empty array to store the realizations.
    x = numpy.empty((N+1,));
    # Initial values of x.
    x[0] = 50;
    x = brownian(x[0], N, dt, delta, out=x[1:]);
    t = numpy.linspace(0.0, T, N+1);
    return [x, t];

def denoise(x):
    #levels = int(log2(len(x)));
    coeffs = pywt.wavedec(x, wavelet='haar');
    threshold = sqrt(2 * log(len(x))); # natural threshold
    new_coeffs = list(numpy.copy(coeffs));
    new_coeffs = [
        pywt.threshold(coeffs[i], threshold, mode='soft')
        for i in range(len(coeffs))
    ]
    ret = pywt.waverec(new_coeffs, wavelet='haar');
    if len(x) % 2 == 0: return ret
    else: return ret[0:-1]


if __name__ == "__main__":
    path = os.path.abspath('BTC-USD-2015-01-01-2017-10-08-daily.csv');
    df = pd.read_csv(path, index_col='time', parse_dates=['time']);

    x = list(df.close.values);
    x = x[1:]; # ignore first row of data

    y = denoise(x);
    y2 = denoise(y);

    mse = sqrt(sklearn.metrics.mean_squared_error(x, y));
    print('MSE = {0}'.format(mse));
    mse2 = sqrt(sklearn.metrics.mean_squared_error(x, y2));
    print('MSE2 = {0}'.format(mse2));

    plt.figure(1);
    df.close.plot();
    plt.show(block=False);
    #plt.show();

    plt.figure(2);
    y_series = pd.Series(y)
    y_series.plot();
    plt.show(block=False);

    plt.figure(3);
    y2_series = pd.Series(y2)
    y2_series.plot();
    plt.show();
