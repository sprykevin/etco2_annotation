import os

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks, find_peaks_cwt, gaussian


out_folder = 'C:/Spry/Code/UCSF2022/rr_annotation/'
fs = 100
window = 30 * fs  # In points
half_window = window // 2


class InteractivePlot:

    def __init__(self, data, fig, ax, subj):
        """ Object to hold data and axis """

        self.start = 0
        self.data = data
        self.fig = fig
        self.ax = ax
        self.subj = subj
        self.results = []
        self.peaks = []
        self.done = False

    def __enter__(self):
        return self

    def __exit__(self, **_):
        self.fig.canvas.flush_events()
        self.fig.canvas.draw()
        self.ax.cla()
        return

    def update(self):

        if not self.done and self.start <= self.data.shape[0] - window:
            # Update the plot
            self.peaks = self.update_plot(self.ax, self.fig, self.data, self.start, self.subj)

        else:
            # Close the class
            self.save()
            self.done = True
            self.__exit__()

    def begin(self):

        self.update()

        while not self.done:
            plt.waitforbuttonpress()

    def save(self):
        pd.DataFrame(self.results).to_csv(os.path.join(out_folder, 'Patient {}.csv'.format(self.subj)))

    def valid(self, _):
        """ Valid button callback """

        self.results.append({
            'subj'     : self.subj,
            'time'     : self.start,
            'valid'    : True,
            'peaks'    : self.peaks,
            'num_peaks': len(self.peaks),
        })

        self.start += window

        self.update()

    def invalid(self, _):
        """ Invalid button callback """

        self.results.append({
            'subj'     : self.subj,
            'time'     : self.start,
            'valid'    : False,
            'peaks'    : self.peaks,
            'num_peaks': len(self.peaks),
        })

        self.start += window

        self.update()

    def prev(self, _):
        """ Previous button callback """

        # Crop out the last result and go back
        self.results = self.results[:-1]

        self.start -= window

        self.update()

    @staticmethod
    def update_plot(ax, fig, data, start, subj):

        try:

            # Signal window for annotation
            s_window = data[start: start + window]

            # Compute peaks on a larger window, then crop
            larger_window = data[max(0, start - half_window): min(start + window + half_window, data.shape[0])]
            peaks = compute_peaks_scipy(larger_window)

            # Crop
            peaks = [p - half_window for p in peaks if (half_window <= p < window + half_window)]

            # Limit peaks to peaks within 20% of the range of the max peak
            if len(peaks) > 0:
                max_peak_val = np.max([s_window[p] for p in peaks])
                signal_range = np.max(s_window) - np.min(s_window)

                peaks = [p for p in peaks if s_window[p] >= max_peak_val - signal_range * 0.25]

            # Set up plot
            ax.cla()
            ax.plot(s_window)
            ax.plot(peaks, s_window[peaks], 'r.', markersize=10)
            for i, p in enumerate(peaks):
                ax.text(p, s_window[p], str(i + 1), fontsize=15, color='black')
            ax.set_title('Subject {}, time = {}s, {} breaths'.format(subj, start / fs, len(peaks)))
            ax.set_ylabel('ETCO2 (mmHg)')

            fig.canvas.flush_events()
            fig.canvas.draw()

            return peaks

        except IndexError:
            ax.cla()
            fig.canvas.flush_events()
            fig.canvas.draw()

            return []


def compute_peaks(sig):
    max_threshold = (np.percentile(sig, 90) + np.percentile(sig, 10)) / 2

    max_idxs = []

    # Find local maxima in the signal
    i = 0
    while i < sig.shape[0]:

        # Detect rise above threshold
        if sig[i] > max_threshold:

            max_value = sig[i]
            max_i = i

            # Find the max in in this rise above threshold
            while i < sig.shape[0] and sig[i] > max_threshold:

                if sig[i] > max_value:
                    # Update local max
                    max_value = sig[i]
                    max_i = i

                i += 1

            # Add found max to list
            max_idxs.append(max_i)

        i += 1

    return max_idxs


def square_wavelet(n, s):
    up = np.ones(s)
    if np.mod(n, 2):
        pad_left, pad_right = (n-s)//2, (n-s)//2
    else:
        pad_left = (n-s)//2 + 1
        pad_right = (n-s)//2

    square = np.hstack((np.zeros(pad_left), up, np.zeros(pad_right)))
    # blur_kernel = np.ones(s//8)
    #
    blur_kernel = gaussian(n//8, std=1)
    wavelet = np.convolve(square, blur_kernel, mode='same')
    wavelet /= np.max(wavelet)
    # wavelet = square
    #
    # print("{} {} {}".format(n, s, len(wavelet)))
    # print(wavelet)

    return wavelet


def compute_peaks_scipy(sig):
    fmin, fmax = 5, 45
    wmin, wmax = 100 * 60 / np.array((fmax, fmin))
    w_space = np.arange(int(wmin), int(wmax), 4)
    # f_space = np.arange(fmin, fmax, 1)[::-1]
    # f_space = np.linspace(fmin, fmax, 100)[::-1]
    # widths = np.array(60 / f_space * fs).astype(int)
    widths = w_space
    # print(widths)

    # width_limits = np.divide(60, (fmax, fmin))*fs
    # widths = np.linspace(width_limits[0], width_limits[1], 20)
    peaks, cwt_data, ridge_lines = find_peaks_cwt(sig, widths=widths, wavelet=square_wavelet)

    return peaks
