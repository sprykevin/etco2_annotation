from PyQt5 import QtGui, QtWidgets, QtCore

import sys, os
import re

import json

import pandas as pd
import numpy as np
from peaks_math import compute_peaks
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from scipy import signal
import copy
# from scipy.signal import find_peaks_cwt, gaussian

out_folder = 'C:/Spry/Code/UCSF2022/rr_annotation/'
fs = 100
window = 30 * fs  # In points
half_window = window // 2


class WindowPlotter(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(WindowPlotter, self).__init__(parent=parent)

        fig = plt.figure(figsize=(14, 4))
        self.canvas = FigureCanvasQTAgg(fig)
        self.fig = self.canvas.figure
        self.ax = fig.add_subplot(111)

        self.layout = QtWidgets.QFormLayout()
        self.layout.addRow(self.canvas)

        self.setLayout(self.layout)

    def plot_signal_and_peaks(self, s_window, peaks, subject, start):
        # Set up plot
        self.ax.cla()
        self.ax.plot(s_window)

        try:
            peaks_and_indices = [(peak.index, idx) for idx, peak in peaks.items() if peak.valid]
            peaks_x, indices = zip(*peaks_and_indices)
            peaks_x = list(peaks_x)


            self.ax.plot(peaks_x, s_window[peaks_x], 'r.', markersize=10)
            for p, i in peaks_and_indices:
                self.ax.text(p, s_window[p], str(i), fontsize=15, color='black')

        except ValueError:
            pass

        self.ax.set_title('Subject {}, time = {}s, {} breaths'.format(subject, start / fs, len(peaks)))
        self.ax.set_ylabel('ETCO2 (mmHg)')

        # self.fig.canvas.flush_events()
        self.fig.canvas.draw()

    def update_plot(self, data, peaks, start, subject):
        try:
            # Signal window for annotation
            s_window = data[start: start + window]

            self.plot_signal_and_peaks(s_window, peaks, subject, start)

        except IndexError:
            self.ax.cla()
            # self.fig.canvas.flush_events()
            self.fig.canvas.draw()


class Plotter(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(Plotter, self).__init__(parent=parent)

        fig = plt.figure(figsize=(14, 2))
        self.canvas = FigureCanvasQTAgg(fig)
        self.fig = self.canvas.figure
        self.ax = fig.add_subplot(111)

        self.layout = QtWidgets.QFormLayout()
        self.layout.addRow(self.canvas)

        self.setLayout(self.layout)

    def update_plot(self, data, start, peaks):
        window_start = max((0, start - 10*window))
        stop = window_start + 21*window

        self.ax.cla()
        self.ax.plot(data)

        try:
            peaks_x = [peak.index for idx, peak in peaks.items() if peak.valid]
            self.ax.plot(peaks_x, data[peaks_x], 'r.', markersize=10)
        except:
            pass

        self.ax.set_xlim((window_start, stop))
        self.ax.set_ylim((0.8 * np.nanmin(data[window_start:stop]), 1.2 * np.nanmax(data[window_start:stop])))
        self.ax.axvline(start)
        self.ax.axvline(start + window)
        self.fig.canvas.draw()


class Buttons(QtWidgets.QWidget):
    BUTTON_WIDTH = 40
    BUTTON_FONT = 'Times'
    BUTTON_FONT_SIZE = 12
    BUTTON_QFONT = QtGui.QFont(BUTTON_FONT, BUTTON_FONT_SIZE)

    def __init__(self, parent=None):
        super(Buttons, self).__init__(parent=parent)

        self.valid = QtWidgets.QPushButton('Valid')
        self.valid.setFont(self.BUTTON_QFONT)
        self.invalid = QtWidgets.QPushButton('Invalid')
        self.invalid.setFont(self.BUTTON_QFONT)
        self.previous = QtWidgets.QPushButton('Previous')
        self.previous.setFont(self.BUTTON_QFONT)

        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.previous)
        self.layout.addWidget(self.invalid)
        self.layout.addWidget(self.valid)

        self.setLayout(self.layout)


class PeakTable(QtWidgets.QWidget):

    def __init__(self, parent=None):
        super(PeakTable, self).__init__(parent=parent)
        self.parent = parent

        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

    def update_with_peaks(self, peaks):
        for i in range(self.layout.count())[::-1]:
            widget = self.layout.itemAt(i).widget()
            if isinstance(widget, QtWidgets.QCheckBox):
                widget.setParent(None)
                self.layout.removeWidget(widget)

        self.setLayout(self.layout)

        for idx, peak in peaks.items():
            row = self.make_peak_row(peak, idx, peak.valid)
            self.layout.addWidget(row)

        self.setLayout(self.layout)

    def make_peak_row(self, peak, idx, valid):
        checkbox = QtWidgets.QCheckBox("{}".format(idx))
        checkbox.setChecked(valid)
        checkbox.stateChanged.connect(lambda: self.parent.checkbox_callback(checkbox, idx))

        return checkbox


class Peak:

    def __init__(self, index, valid):
        self.index = index
        self.valid = valid


class Window(QtWidgets.QWidget):
    INSTRUCTION_FONT = QtGui.QFont('Arial', 12)

    def __init__(self, subject, data, app):
        super(Window, self).__init__(parent=None)

        self.subject = subject
        self.data = data
        self.app = app

        self.start = 0
        self.peaks = {}
        self.all_peaks = {}
        self.results = []

        self.all_peaks = {}

        self.instructions = QtWidgets.QLabel()
        self.instructions.setText(
            "Valid window criteria: \n 1. The window contains only breaths. \n 2. The approximate end of each "
            "exhalation (maximum value) is labeled."
        )
        self.instructions.setFont(self.INSTRUCTION_FONT)

        self.app_usage_instructions = QtWidgets.QLabel()
        self.app_usage_instructions.setText(
            "To navigate signal windows, use the provided buttons. For convenience, the buttons are also mapped to "
            "the keyboard. Remove incorrect annotations by deselecting the appropriate label using the checkboxes."
        )
        self.app_usage_instructions.setFont(self.INSTRUCTION_FONT)
        self.app_usage_instructions.setWordWrap(True)

        self.buttons = Buttons(self)
        self.buttons.valid.clicked.connect(self.valid)
        self.buttons.invalid.clicked.connect(self.invalid)
        self.buttons.previous.clicked.connect(self.previous)


        instructions_row = QtWidgets.QHBoxLayout()
        instructions_row.addWidget(self.instructions)
        instructions_row.addWidget(self.app_usage_instructions)
        # instructions_row.addWidget(self.buttons)

        self.window_plotter = WindowPlotter(self)
        self.window_plotter.fig.canvas.mpl_connect('button_press_event', self.canvas_click)
        self.plotter = Plotter(self)


        self.table = PeakTable(self)
        # self.table.setFixedHeight(100)

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)
        self.scroll.setMinimumSize(200, 100)
        self.scroll.setWidget(self.table)

        plots = QtWidgets.QVBoxLayout()
        plots.addWidget(self.window_plotter)
        plots.addWidget(self.plotter)

        table_and_buttons = QtWidgets.QHBoxLayout()
        table_and_buttons.addWidget(self.buttons)

        plots_and_table = QtWidgets.QHBoxLayout()
        plots_and_table.addLayout(plots)
        plots_and_table.addWidget(self.scroll)

        self.layout = QtWidgets.QFormLayout()
        self.layout.addRow(instructions_row)
        self.layout.addRow(plots_and_table)
        self.layout.addRow(table_and_buttons)

        self.setLayout(self.layout)

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_F:
            self.valid()
        elif event.key() == QtCore.Qt.Key_D:
            self.invalid()
        elif event.key() == QtCore.Qt.Key_S:
            self.previous()

    def canvas_click(self, event):
        # Left mouse click
        if event.button == 1 and 0 <= event.xdata <= window:
            new_idx = len(self.peaks.keys()) + 1
            self.peaks[new_idx] = Peak(int(event.xdata), True)
            self.window_plotter.update_plot(self.data, self.peaks, self.start, self.subject)
            self.table.update_with_peaks(self.peaks)
            # self.window_plotter.update_plot(self.data, self.peaks, self.start, self.subject)
            # print('{} {}'.format(event.button, event.x))

    def checkbox_callback(self, checkbox, idx):
        self.peaks[idx].valid = checkbox.isChecked()
        self.window_plotter.update_plot(self.data, self.peaks, self.start, self.subject)
        # print(self.peaks)

    def update_results(self, valid_window):
        valid_peaks = sorted([peak.index for key, peak in self.peaks.items() if peak.valid], reverse=False)

        self.results.append({
            'subj'     : self.subject,
            'time'     : self.start,
            'valid'    : valid_window,
            'peaks'    : valid_peaks,
            'num_peaks': len(valid_peaks),
        })

    def update_peaks(self, valid):
        for key, peak in self.peaks.items():
            index_absolute = peak.index + self.start
            if peak.valid and valid:
                self.all_peaks[index_absolute] = Peak(index_absolute, valid)
            else:
                try:
                    self.all_peaks.pop(index_absolute)
                except KeyError:
                    pass

    def pop_peaks(self):
        for key in list(self.all_peaks.keys())[::-1]:
            peak = self.all_peaks[key]
            if peak.index >= self.start:
                self.all_peaks.pop(peak.index)

    def valid(self):
        """ Valid button callback """

        self.update_results(True)
        self.update_peaks(True)

        self.start += window

        self.clear_peaks()
        self.update()

    def invalid(self):
        """ Invalid button callback """

        self.update_results(False)
        self.update_peaks(False)

        self.start += window

        self.clear_peaks()
        self.update()

    def previous(self):
        """ Previous button callback """

        # Crop out the last result and go back
        self.results = self.results[:-1]

        self.start -= window
        self.pop_peaks()

        self.clear_peaks()
        self.update()

    def clear_peaks(self):
        # Clear peaks
        self.last_peaks = copy.deepcopy(self.peaks)
        self.peaks = {}

    def update(self):
        print(list(self.all_peaks.keys())[-10:])

        if self.start <= self.data.shape[0] - window:
            peaks = self.get_peaks(self.data, self.start)

            for idx, peak in enumerate(peaks):
                self.peaks[idx + 1] = Peak(peak, True)
                # self.all_peaks.append(Peak(self.start + peak, True))
                # self.all_peaks[]

            # Update the plot
            self.window_plotter.update_plot(self.data, self.peaks, self.start, self.subject)
            self.plotter.update_plot(self.data, self.start, self.all_peaks)

            self.table.update_with_peaks(self.peaks)

        else:
            # Close the class
            self.save()
            self.app.quit()
            self.close()

    def get_peaks(self, data, start):
        order = 2
        # Signal window for annotation
        s_window = data[start: start + window]

        # Compute peaks on a larger window, then crop
        larger_window = data[max(0, start - half_window): min(start + window + half_window, data.shape[0])]
        # larger_window = detrend_polynomial(larger_window, order=order).flatten()
        peaks = compute_peaks(larger_window)

        # Crop
        peaks = [p - half_window for p in peaks if (half_window <= p < window + half_window)]

        # Limit peaks to peaks within 20% of the range of the max peak
        if len(peaks) > 0:
            max_peak_val = np.max([s_window[p] for p in peaks])
            signal_range = np.max(s_window) - np.min(s_window)

            peaks = [p for p in peaks if s_window[p] >= max_peak_val - signal_range * 0.25]

        return peaks

    def save(self):
        pd.DataFrame(self.results).to_csv(os.path.join(out_folder, 'Subject {}.csv'.format(self.subject)))


class WindowValidator(Window):

    def __init__(self, subject, data, app, peaks_data):
        super(WindowValidator, self).__init__(subject, data, app)

        self.peaks_data = peaks_data

    def get_peaks(self, data, start):
        valid = self.peaks_data[start]['valid']
        if valid:
            return json.loads(self.peaks_data[start]['peaks'])
        else:
            return []


def process_data():
    in_folder = 'C:/Spry/Code/UCSF2022/reference/'

    # Find gt files
    gt_files = []

    for path, subdirs, files in os.walk(in_folder):
        for name in files:
            if 'Raw' in name and 'Subject' in name:
                gt_files.append(os.path.join(path, name))

    print(gt_files)

    for f in gt_files:
        app = QtWidgets.QApplication.instance()
        if not app:
            app = QtWidgets.QApplication(sys.argv)


        # Define data for this subject
        data = pd.read_csv(f, delimiter='\t', skiprows=2, names=['etco2', 'eto2', 'bp'])['etco2'].to_numpy()

        # Basic filter for noise
        kernel = signal.firwin(401, (2/60, 45/60), fs=fs, pass_zero=False, window='hamming')
        data = np.convolve(data, kernel, mode='same')
        subject = int(re.findall(r'.*?\Subject #(.*) Raw.*', f)[0])

        window = Window(subject, data, app)
        window.show()
        window.update()
        app.exec_()

        # window.plotter.ax.plot(data)
        # window.window_plotter.ax.plot(data[:fs*30])


def display_processed_data():
    in_folder = 'C:/Spry/Code/UCSF2022/reference/'
    annotation_folder = 'C:/Spry/Code/UCSF2022/rr_annotation/'

    # Find gt files
    gt_files = []

    for path, subdirs, files in os.walk(in_folder):
        for name in files:
            if 'Raw' in name and 'Subject' in name:
                gt_files.append(os.path.join(path, name))

    print(gt_files)

    for f in gt_files:
        app = QtWidgets.QApplication.instance()
        if not app:
            app = QtWidgets.QApplication(sys.argv)


        # Define data for this subject
        data = pd.read_csv(f, delimiter='\t', skiprows=2, names=['etco2', 'eto2', 'bp'])['etco2'].to_numpy()

        # Basic filter for noise
        kernel = signal.firwin(401, (2/60, 45/60), fs=fs, pass_zero=False, window='hamming')
        data = np.convolve(data, kernel, mode='same')
        subject = int(re.findall(r'.*?\Subject #(.*) Raw.*', f)[0])

        peaks_data = pd.read_csv('{}/Subject {}.csv'.format(annotation_folder, subject))
        peaks_data = peaks_data.set_index('time')[['peaks', 'valid']].to_dict('index')

        window = WindowValidator(subject, data, app, peaks_data)
        window.show()
        window.update()
        app.exec_()


if __name__ == '__main__':
    display_processed_data()

