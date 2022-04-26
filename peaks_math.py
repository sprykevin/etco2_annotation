import numpy as np
from scipy.signal import find_peaks_cwt


def least_squares_fit(y, A):
    return np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y.T).flatten()


def linear_regression(x, y):
    A = np.concatenate((np.ones(x.shape), x), axis=0).T

    return least_squares_fit(y, A), x, y


def polynomial_regression(x, y, order):
    # A0 = np.concatenate((np.ones(x.shape), x, np.power(x, 2)), axis=0).T
    A = np.concatenate([np.power(x, p) for p in np.arange(order + 1)], axis=0).astype(float).T
    return least_squares_fit(y, A), x, y


def detrend_polynomial(sig, order):
    x = np.arange(len(sig)).reshape(1, -1)
    coeffs, x, y = polynomial_regression(x, sig, order)
    trend = np.sum(np.array([coeffs[p] * np.power(x, p) for p in np.arange(1, order+1)]), axis=0).flatten()
    return np.subtract(sig, trend)


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


def compute_peaks_scipy(sig):
    fmin, fmax = 5, 40
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