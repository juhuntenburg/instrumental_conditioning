import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from scipy import stats

data_dir = '/home/julia/data/ict/'
remove = 1000

for mouse in ['SHA', 'TAY', 'UUU', 'VVV', 'WEY', 'YOU']:

    # Load data for each mouse
    df = pd.read_pickle(data_dir+'mice/{}.pkl'.format(mouse))

    # Loop over sessions (i.e. days)
    for day in np.unique(df['day']):
        raw_signal = df[(df['day']==day)]['gpmt']
        raw_reference = df[(df['day']==day)]['rpmt']

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

        # Fit Gaussian mixture model to reference signal of entire sessions
        gmm = GaussianMixture(n_components=2, covariance_type='spherical')
        fit = gmm.fit(all_reference.reshape(-1,1))
        predict = gmm.predict(all_reference.reshape(-1,1))

        # Calculate crossing between the two Gaussians

        # Calculate linear regression only based on



    all_signal = np.append(all_signal, delta_signal[remove:-remove])
    all_reference = np.append(all_reference, delta_reference[remove:-remove])

    all_signal = np.append(all_signal, delta_signal[remove:-remove])
    all_reference = np.append(all_reference, delta_reference[remove:-remove])

        for trial in range(len(raw_signal)):

            # Smooth signal and reference with Gaussian filter
            smooth_signal = gaussian_filter1d(raw_signal.iloc[trial],
                                              sigma=sigma, output=np.float64)
            smooth_reference = gaussian_filter1d(raw_reference.iloc[trial],
                                                 sigma=sigma, output=np.float64)

            # Remove beginning and end to avoid filterint artefacts
            short_signal = smooth_signal[remove:-remove]
            short_reference = smooth_reference[remove:-remove]

            # Fit and remove baselines
            predict_signal = airPLS(short_signal, lambda_=lam, itermax=50)
            predict_reference = airPLS(short_reference, lambda_=lam, itermax=50)

            signal = short_signal - predict_signal
            reference = short_reference - predict_reference

            # Standardize
            z_reference = (reference - np.median(reference)) / np.std(reference)
            z_signal = (signal - np.median(signal)) / np.std(signal)

            # Fit reference to signal using linear regression and remove
            lin = LinearRegression()
            n = len(z_reference)
            lin.fit(z_reference.reshape(n,1), z_signal.reshape(n,1))
            z_reference_fitted = lin.predict(z_reference.reshape(n,1)).reshape(n,)
            zdFF = (z_signal - z_reference_fitted)

            all_zdFF.append(zdFF)

        df['gpmt_corr'] = all_zdFF
        df.to_pickle(data_dir+'mice/{}.pkl'.format(mouse))
