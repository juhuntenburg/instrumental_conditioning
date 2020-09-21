import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression
from photometry_functions import airPLS

data_dir = '/home/julia/data/ict/'
remove = 1000
sigma=100
lam = 4e5


for mouse in ['UUU', 'VVV', 'WEY', 'YOU']: #'SHA', 'TAY'

    # Load data for each mouse
    df = pd.read_pickle(data_dir+'mice/{}.pkl'.format(mouse))
    all_corr = []

    # Loop over sessions (i.e. days)
    for day in np.unique(df['day']):
        raw_signal = df[(df['day']==day)]['gpmt']
        raw_reference = df[(df['day']==day)]['rpmt']

        print(mouse, day)

        list_signal = []
        list_reference = []
        all_signal = np.empty(0)
        all_reference = np.empty(0)
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

            list_signal.append(z_signal)
            list_reference.append(z_reference)

            all_signal = np.append(all_signal, z_signal)
            all_reference = np.append(all_reference, z_reference)

        # Fit reference to signal using linear regression and remove
        lin = LinearRegression()
        n = len(all_reference)
        model = lin.fit(all_reference.reshape(n,1), all_signal.reshape(n,1))
        all_reference_fitted = lin.predict(all_reference.reshape(n,1)).reshape(n,)

        # Clean up all trials using linear fit coefficients
        a = model.intercept_[0]
        b = model.coef_[0,0]

        for trial in range(len(raw_signal)):
            all_corr.append(list_signal[trial] - a*list_reference[trial] - b)

    df['gpmt_corr_persession'] = all_corr
    df.to_pickle(data_dir+'mice/{}.pkl'.format(mouse))
