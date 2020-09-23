import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LinearRegression

data_dir = '/home/julia/data/ict/'
mice = ['YOU', 'TAY', 'UUU', 'VVV', 'WEY']
#mice = ['SHA']
days = [-3,-2,-1,0,1,2]
#days = [-3,-2,-1,0]
remove = 1000


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
    corr_signal = []

    for day in range(len(days)):
        print(mouse, days[day])
        raw_signal = df[(df['day']==days[day])]['gpmt']
        raw_reference = df[(df['day']==days[day])]['rpmt']
        odor_start = df[(df['day']==days[day])]['odor_start']

        all_signal = np.empty(0)
        all_reference = np.empty(0)

        # Smooth and F/F0 for each trial
        for trial in range(len(raw_signal)):
            smooth_signal = gaussian_filter1d(raw_signal.iloc[trial], sigma=100, axis=0, output=np.float64)
            smooth_reference = gaussian_filter1d(raw_reference.iloc[trial], sigma=100, axis=0, output=np.float64)

            F0_signal = np.mean(smooth_signal[(odor_start.iloc[trial]-1000):odor_start.iloc[trial]])
            F0_reference = np.mean(smooth_reference[(odor_start.iloc[trial]-1000):odor_start.iloc[trial]])
            delta_signal = (smooth_signal - F0_signal) / F0_signal
            delta_reference = (smooth_reference - F0_reference) / F0_reference

            all_signal = np.append(all_signal, delta_signal[remove:-remove])
            all_reference = np.append(all_reference, delta_reference[remove:-remove])

        # Fit linear regression on full session
        lin = LinearRegression()
        n = len(all_reference)
        model = lin.fit(all_reference.reshape(n,1), all_signal.reshape(n,1))
        reference_fitted = lin.predict(all_reference.reshape(n,1)).reshape(n,)
        a = model.coef_[0,0]
        b = model.intercept_[0]

        # Create QC figure
        plot_signal = (all_signal - reference_fitted)
        idx=np.arange(0,all_signal.shape[0],100)

        fig = plt.figure(figsize=(10, 10))
        ax1 = fig.add_subplot(311)
        ax1.plot(all_reference[idx],all_signal[idx],'k.', linewidth=0)
        ax1.plot(all_reference[idx],reference_fitted[idx], color='darkgrey',
                 linewidth=2.5)
        ax1.set_xlabel('Red PMT')
        ax1.set_ylabel('Green PMT')

        ax2 = fig.add_subplot(312)
        ax2.plot(all_signal,'darkgreen', label='Green PMT')
        ax2.plot(reference_fitted,'darkred', label='Fitted Red PMT')
        ax2.set_xlabel('sec')

        ax3 = fig.add_subplot(313)
        ax3.plot(plot_signal,'limegreen')
        ax3.set_xlabel('sec')

        plt.tight_layout()
        sns.despine()
        plt.savefig(data_dir+'qc/dFF0/{}_{}.png'.format(mouse, day))

        # Remove reference_from each trial
        for trial in range(len(raw_signal)):
            smooth_signal = gaussian_filter1d(raw_signal.iloc[trial], sigma=100, axis=0, output=np.float64)
            smooth_reference = gaussian_filter1d(raw_reference.iloc[trial], sigma=100, axis=0, output=np.float64)
            F0_signal = np.mean(smooth_signal[(odor_start.iloc[trial]-1000):odor_start.iloc[trial]])
            F0_reference = np.mean(smooth_reference[(odor_start.iloc[trial]-1000):odor_start.iloc[trial]])
            delta_signal = (smooth_signal - F0_signal) / F0_signal
            delta_reference = (smooth_reference - F0_reference) / F0_reference

            corr_signal.append(delta_signal - a*delta_reference - b)


    df['gpmt_corr'] = corr_signal
    df.drop('gpmt', axis=1)
    df.drop('rpmt', axis=1)
    df.to_pickle(data_dir+'mice/{}_corr_dFF0.pkl'.format(mouse))
