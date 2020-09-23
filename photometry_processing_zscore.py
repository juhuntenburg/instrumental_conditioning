import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from scipy.special import legendre

data_dir = '/home/julia/data/ict/'
mice = ['YOU', 'TAY', 'UUU', 'VVV', 'WEY']
#mice = ['SHA']
days = [-3,-2,-1,0,1,2]
#days = [-3,-2,-1,0]


for mouse in mice:
    # Load data for each mouse
    df = pd.read_pickle(data_dir+'mice/{}.pkl'.format(mouse)).filter(['dataset',
                                                                      'day',
                                                                      'type',
                                                                      'odor',
                                                                      'performance',
                                                                      'gpmt',
                                                                      'rpmt',
                                                                      'odor_start',
                                                                      'iti_start'])
    df = df[df['day'].isin(days)]
    corr_series = []

    for day in range(len(days)):
        print(mouse, days[day])

        signal_series = df[(df['day']==days[day])]['gpmt']
        reference_series = df[(df['day']==days[day])]['rpmt']

        raw_signal = np.empty(0)
        for trial in signal_series:
            raw_signal = np.append(raw_signal, trial)

        raw_reference = np.empty(0)
        for trial in reference_series:
            raw_reference = np.append(raw_reference, trial)

        smooth_signal = gaussian_filter1d(raw_signal, sigma=100, output=np.float64)
        smooth_reference = gaussian_filter1d(raw_reference, sigma=100, output=np.float64)

        poly_1 = legendre(1)(np.linspace(-1,1,smooth_signal.shape[0]))
        poly_2 = legendre(2)(np.linspace(-1,1,smooth_signal.shape[0]))
        poly_3 = legendre(3)(np.linspace(-1,1,smooth_signal.shape[0]))
        X = np.vstack((poly_1, poly_2, poly_3)).T

        lin_signal = LinearRegression()
        model_signal = lin_signal.fit(X, smooth_signal)
        predict_signal = lin_signal.predict(X)

        lin_reference = LinearRegression()
        model_reference = lin_reference.fit(X, smooth_reference)
        predict_reference = lin_reference.predict(X)

        signal = smooth_signal - predict_signal
        reference = smooth_reference - predict_reference

        z_reference = (reference - np.median(reference)) / np.std(reference)
        z_signal = (signal - np.median(signal)) / np.std(signal)

        lin = LinearRegression()
        n = len(z_reference)
        model = lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))
        reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)
        corr_signal = (z_signal - reference_fitted)

        # Create QC figure
        idx=np.arange(0,z_signal.shape[0],100)

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(311)
        ax1.plot(z_reference[idx],z_signal[idx],'k.', linewidth=0)
        ax1.plot(z_reference[idx],reference_fitted[idx], color='darkgrey',
                 linewidth=2.5)
        ax1.set_xlabel('Red PMT')
        ax1.set_ylabel('Green PMT')

        ax2 = fig.add_subplot(312)
        ax2.plot(z_signal,'darkgreen', label='Green PMT')
        ax2.plot(reference_fitted,'darkred', label='Fitted Red PMT')
        ax2.set_xlabel('sec')

        ax3 = fig.add_subplot(313)
        ax3.plot(corr_signal,'limegreen')
        ax3.set_xlabel('sec')

        plt.tight_layout()
        sns.despine()
        plt.savefig(data_dir+'qc/zscore/{}_{}.png'.format(mouse, days[day]))
        plt.close()

        # Split session into trials again and append to series
        idx = 0
        for trial in signal_series:
            corr_series.append(corr_signal[idx:idx+trial.shape[0]])
            idx += trial.shape[0]

    df['gpmt_corr'] = corr_series
    df.drop('gpmt', axis=1)
    df.drop('rpmt', axis=1)
    df.to_pickle(data_dir+'mice/{}_corr_zscore.pkl'.format(mouse))
