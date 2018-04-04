import numpy as np
import pandas as pd
import matplotlib as plt
from pandas.plotting import autocorrelation_plot

from statsmodels.tsa.stattools  import adfuller, pacf


def mean_predictions(predicted):
    """
    Calculate the mean of predictions that overlaps. This is donne mostly to be able to plot what the model is doing.
    -------------------------------------------------------
    Args:
        predicted : numpy array
            Numpy array with shape (Number points to predict - prediction length -1, predictions length)

    -------------------------------------------------------
    return:
        predictions_mean : list
            list with len of number to predict where each position is the mean of all predictions to that step
    """

    array_global = [[] for _ in range((predicted.shape[0] + predicted.shape[1]))]

    for i in range(predicted.shape[0]):
        for l, value in enumerate(predicted[i]):
            array_global[i + l].append((float(value)))

    predictions_mean = []

    for i in range(len(array_global) - 1):
        predictions_mean.append(np.array(array_global[i]).mean())

    return predictions_mean


def differentiate_timeseries(timeseries, diff_lag=1):
    """
    Differentiate a time-series signal.
    -------------------------------------------------------
    Args:
        timeseries : pd.DataFrame or pd.Series
            Pandas Series with time series to be differentiated
        diff_lag: int
            Time difference to use
    -------------------------------------------------------
    return:
        predictions_mean : pd.Series
            Differentiated time series

    """
    diff = timeseries.diff(diff_lag)
    return diff.fillna(0)


def logarithmic_scaling(timeseries, flag=True):
    """
    Logarithmic scaling of time-series
    -------------------------------------------------------
    Args:
        timeseries : pd.Series
            Pandas Series to apply transformation
        flag: boolean
            If True use np.log(x + 1) instead of np.log(x). This is useful if any point is equal to zero

    -------------------------------------------------------
    return:
        timeseries_log : pd.Series
            Transformed time-series
    """
    if flag:
        timeseries_log = timeseries.apply(lambda x: np.log(x + 1))
    else:
        timeseries_log = timeseries.apply(lambda x: np.log(x))

    return timeseries_log.fillna(0)


def test_stationarity(timeseries, windown_size=7, plot_graph=True, adfuller_test=False, figure_size=(40, 20)):
    """
    Calculate adfuller test to calculate time series stationary properties. If plot_graph is True then it will be
    plotted the true signal, the rolling mean series and the ACF and ACPF plots
    -------------------------------------------------------
    Args:
        timeseries : pd.Series
            Time-series to test
        windown_size: int
            Window size to plot rolling mean
        plot graph: boolean
            If true plot graph. False do nothing
        adfuller_test: bolean
            If to perform the adfuller test or not

    """

    if plot_graph == True:
        fig, axs = plt.subplots(2, 2, figsize=(40, 20))

        timeseries.plot(ax=axs[0, 0], title="Original Time-Series")

        timeseries.rolling(window=windown_size, center=False).mean().plot(ax=axs[0, 1],
                                                                          title='Time-Series with rolling mean of {}'.format(
                                                                              windown_size));

        autocorrelation_plot(timeseries, ax=axs[1, 0]).set_title('Auto-correlation function plot')

        pd.Series(pacf(timeseries)).plot(ax=axs[1, 1], title='Partial auto-correlation function plot');

    if adfuller_test == True:
        print('Results of Dickey-Fuller Test:')

        dftest = adfuller(timeseries, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value

        print(dfoutput)


def adfuller_dataframe(df):
    """
    Calculate the adfuller test for a pd.Dataframe with several time-series on each row
    -------------------------------------------------------
    Args:
        df : pd.DataFrame
            Dataframe with time-series

    -------------------------------------------------------
    return:
        dataframe : pd.DataFrame
            Dataframe with statistical values from adfuller test for each time-series
    """
    dataframe = pd.DataFrame(index=['Test Statistic',
                                    'p-value',
                                    '#Lags Used',
                                    'Number of Observations Used',
                                    'Critical Value - 1%',
                                    'Critical Value - 5%',
                                    'Critical Value - 10%'],
                             columns=df.columns)

    for col in df:
        series = df[col].fillna(0)
        dftest = adfuller(series, autolag='AIC')

        dfoutput = pd.Series(dftest[0:4],
                             index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

        for key, value in dftest[4].items():
            dfoutput['Critical Value (%s)' % key] = value

        dataframe[col] = dfoutput.values

    return dataframe.round(3)
