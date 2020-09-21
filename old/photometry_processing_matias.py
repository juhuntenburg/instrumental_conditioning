import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = '/home/julia/data/ict/'
remove = 1000

for mouse in ['VVV', 'WEY', 'YOU']: #'SHA', 'TAY', 'UUU'

    # Load data for each mouse
    df = pd.read_pickle(data_dir+'mice/{}.pkl'.format(mouse))
    all_corr = []

    # Loop over sessions (i.e. days)
    for day in np.unique(df['day']):
        raw_signal = df[(df['day']==day)]['gpmt']
        raw_reference = df[(df['day']==day)]['rpmt']
        odor_start = df[(df['day']==day)]['odor_start']

        print(mouse, day)

        all_signal = np.empty(0)
        all_reference = np.empty(0)

        # For each trial, smooth and calculate deltaF/F0
        for trial in range(len(raw_signal)):
            smooth_signal = gaussian_filter1d(raw_signal.iloc[trial], sigma=100, axis=0, output=np.float64)
            smooth_reference = gaussian_filter1d(raw_reference.iloc[trial], sigma=100, axis=0, output=np.float64)

            F0_signal = np.mean(smooth_signal[(odor_start.iloc[trial]-1000):odor_start.iloc[trial]])
            F0_reference = np.mean(smooth_reference[(odor_start.iloc[trial]-1000):odor_start.iloc[trial]])
            delta_signal = (smooth_signal - F0_signal) / F0_signal
            delta_reference = (smooth_reference - F0_reference) / F0_reference

            all_signal = np.append(all_signal, delta_signal[remove:-remove])
            all_reference = np.append(all_reference, delta_reference[remove:-remove])

        # Fit Gaussian mixture model to reference signal of entire sessions
        gmm = GaussianMixture(n_components=2, covariance_type='spherical')
        fit = gmm.fit(all_reference.reshape(-1,1))
        predict = gmm.predict(all_reference.reshape(-1,1))

        # Calculate linear regression, removing the values of the
        # zero centered Gaussian
        means = np.abs(fit.means_.flatten())
        noise_idx = np.where(means==np.amin(means))[0][0]
        fit_idx = np.where(means==np.amax(means))[0][0]

        fit_reference = all_reference[predict==fit_idx]
        fit_signal = all_signal[predict==fit_idx]

        lin = LinearRegression()
        model = lin.fit(fit_reference.reshape(-1,1), fit_signal.reshape(-1,1))
        reference_fitted = lin.predict(all_reference.reshape(-1,1))

        # Create plot for QC
        non_fit_signal = all_signal[predict==noise_idx]
        non_fit_reference = all_reference[predict==noise_idx]
        idx_a=np.arange(0,fit_signal.shape[0],100)
        idx_b=np.arange(0,non_fit_signal.shape[0],100)

        fig = plt.figure(figsize=(10, 8))
        ax1 = fig.add_subplot(211)
        sns.despine(ax=ax1, left=True, bottom=True)
        sns.distplot(all_reference, color='black', hist=True, ax=ax1)
        ax2 = fig.add_subplot(212)
        ax2.plot(fit_reference[idx_a],fit_signal[idx_a],color='black', marker='.',linewidth=0)
        ax2.plot(non_fit_reference[idx_b],non_fit_signal[idx_b],color='grey', marker='.', linewidth=0)
        ax2.plot(all_reference,reference_fitted, color='darkred',linewidth=3)
        sns.despine()
        plt.savefig(data_dir+'qc/matias/{}_{}.png'.format(mouse, day))
        plt.close()

        # Clean up all trials using linear fit coefficients
        a = model.intercept_[0]
        b = model.coef_[0,0]

        for trial in range(len(raw_signal)):
            smooth_signal = gaussian_filter1d(raw_signal.iloc[trial], sigma=100, axis=0, output=np.float64)
            smooth_reference = gaussian_filter1d(raw_reference.iloc[trial], sigma=100, axis=0, output=np.float64)

            F0_signal = np.mean(smooth_signal[(odor_start.iloc[trial]-1000):odor_start.iloc[trial]])
            F0_reference = np.mean(smooth_reference[(odor_start.iloc[trial]-1000):odor_start.iloc[trial]])
            delta_signal = (smooth_signal - F0_signal) / F0_signal
            delta_reference = (smooth_reference - F0_reference) / F0_reference

            corr_signal = delta_signal - a*delta_reference - b

            # Standardize
            z_corr = (corr_signal - np.median(corr_signal)) / np.std(corr_signal)
            all_corr.append(z_corr)

    df['gpmt_corr_matias'] = all_corr
    df.to_pickle(data_dir+'mice/{}.pkl'.format(mouse))
