import pandas as pd
import numpy as np
import os
import re

from gui import InteractivePlot, fs
from scipy import signal
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

in_folder = 'C:/Spry/Code/UCSF2022/reference/'


def run_one_subject(data, subj):
    # Set up figure, axes
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)

    # Set up buttons
    axprev = plt.axes([0.59, 0.05, 0.1, 0.075])
    axinvalid = plt.axes([0.7, 0.05, 0.1, 0.075])
    axvalid = plt.axes([0.81, 0.05, 0.1, 0.075])

    button_axes = {'axprev': axprev, 'axinvalid': axinvalid, 'axvalid': axvalid}

    plt.ion()
    plt.show()

    interactive_plots = InteractivePlot(data, fig, ax, subj)

    bvalid = Button(button_axes['axvalid'], 'Valid')
    bvalid.on_clicked(interactive_plots.valid)
    binvalid = Button(button_axes['axinvalid'], 'Invalid')
    binvalid.on_clicked(interactive_plots.invalid)
    bprev = Button(button_axes['axprev'], 'Prev')
    bprev.on_clicked(interactive_plots.prev)

    interactive_plots.begin()

    plt.close()


def process_data():

    # Find gt files
    gt_files = []

    for path, subdirs, files in os.walk(in_folder):
        for name in files:
            if 'Raw' in name and 'Subject' in name:
                gt_files.append(os.path.join(path, name))

    print(gt_files)

    for f in gt_files[2:]:
        print(f)

        # Define data for this subject
        data = pd.read_csv(f, delimiter='\t', skiprows=2, names=['etco2', 'eto2', 'bp'])['etco2'].to_numpy()

        # Basic filter for noise
        # kernel = signal.firwin(401, (2/60, 45/60), fs=fs, pass_zero=False, window='hamming')
        # data = np.convolve(data, kernel, mode='same')
        subj = int(re.findall(r'.*?\Subject #(.*) Raw.*', f)[0])

        run_one_subject(data, subj)


if __name__ == "__main__":

    process_data()

    plt.close()
