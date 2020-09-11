import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from photometry_functions import airPLS

data_dir = '/home/julia/data/ict/'
remove = 1000
sigma=100
lam = 4e5

for mouse in ['SHA', 'TAY', 'UUU', 'VVV', 'WEY', 'YOU']:

    # Load data for each mouse
    df = pd.read_pickle(data_dir+'mice/{}.pkl'.format(mouse))
    raw_signal = df['gpmt']
    raw_reference = df['rpmt']

    print(mouse)

    all_zdFF = []
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
