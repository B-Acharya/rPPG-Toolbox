#!/usr/bin/env python3
"""
This script processes the raw signal and video files to generate cropped videos based on the trigger
@author: Bhargav Acharya
@data: 22.08.2023

Usage: python main.py
for additional info use python3 process-raw-data.py --help
"""

import json
import click
import numpy as np
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
import h5py



class sensor_data():

    def __init__(self, sensor_path, videoPath, order, plot_trigger=False, savePath = None, plot=False):

        #default ordering of folders
        defaultOrder = ['LowHR-Bright', 'LowHR-Dark', 'HighHR-Dark', 'HighHR-Bright']
        self.folderStructure = {o: str(i) for i, o in enumerate(defaultOrder)}

        self.videoPath = Path(videoPath)
        if savePath == None:
            self.savePath = self.videoPath.parent
        else:
            self.savePath = savePath

        for f in self.folderStructure.values():
            (self.savePath / f).mkdir(exist_ok=True)


        #extract the information from txt file
        header, data = self._process_sensor_file(sensor_path)

        #order of the videos that are recorded
        self.order = order

        #number of channels present in the data
        self.channels = header['sensor']
        self.labels = header['label']

        #windowlength in seconds
        self.windowLength = 60
        self.numOfWindows = 4

        #sampling rate of the data
        self.sampling_rate = header['sampling rate']

        #number of samples in each window
        self.samples = self.windowLength * self.sampling_rate

        #process the data
        self.data_dict = self._process_data(data, header)
        print(self.data_dict)

        #the trigger signal label name
        self.trigger_signal = "CUSTOM/0.5/1.0/V"
        self.indices = self._find_crop_indices()

        #added just to process false triggers from the pilot study
        # self.indices = self.indices[1:]

        if plot_trigger:
            plt.plot(self.data_dict[self._get_label(self.trigger_signal)])
            plt.show()
            plt.close()

        #split the data based on the timestamps from the trigger signal
        self._data_split()
        self._plot()
        #crop the video and save them
        self._split_video()

    def _split_video(self):
        for i, order in enumerate(self.order):
            if i == 0:
                subprocess.run(['ffmpeg', '-ss', '0', '-t', '60', '-i',
                            str(self.videoPath), '-c', 'copy', str(self.savePath) + f"/{self.folderStructure[order]}/" + order + ".MOV"])

            else:
                timestamps = self._to_seconds(self.indices[i] - self.indices[0])
                print(timestamps)
                subprocess.run(['ffmpeg', '-ss', f'{timestamps}', '-t', '60', '-i', str(self.videoPath), '-c', 'copy',
                                str(self.savePath) + f"/{self.folderStructure[order]}/" + order + ".MOV"])
            self._save_to_hdf5(order)

    def _save_to_hdf5(self, order):
        data = self.split_data[order]
        hf = h5py.File(str(self.savePath) + f"/{self.folderStructure[order]}/data.hdf5", "w")
        hf['ecg'] = data['ECG']
        hf['bvp'] = data['BVP']
        hf['respiration'] = data['RESPIRATION']
        hf['ecg'].attrs['sampling rate'] = self.sampling_rate
        hf.close()


    def _plot(self):
        for key in self.split_data.keys():
            data = self.split_data[key]
            fig, axs = plt.subplots(4, figsize= (16, 16))
            for i, in_key in enumerate(data.keys()):
                x = data[in_key]
                if in_key == "timestamps":
                    continue
                axs[i].plot(x)
                axs[i].set_title(in_key)
            fig.suptitle(key)
            plt.plot()
            plt.show()
            plt.close(fig)
            plt.savefig(str(self.savePath / f"{key}_plot.png"))

    def _data_split(self):
        self.split_data = dict()
        for i, order in enumerate(self.order):
            temp = dict()
            for key in self.data_dict.keys():
                if key=="timestamps":
                    temp[key] = self._crop_signal(i, self.data_dict[key])
                else:
                    temp[self._get_channel(key)] = self._crop_signal(i, self.data_dict[key])
            self.split_data[order] = temp

    def _to_seconds(self, index):
        return index//self.sampling_rate

    def _crop_signal(self, orderNum, signal):
        index = self.indices[orderNum]
        signal = signal[index:index + self.samples]
        return signal

    def _find_crop_indices(self):
        signal = self.data_dict[self._get_label(self.trigger_signal)]
        threshold = (min(signal) + max(signal))//2
        signal[signal < threshold] = 0
        signal[signal > threshold] = 1

        #check if there is a drop inbetween the signal can be removed
        prev = 0
        index = []
        for i in range(1,len(signal)-1):
            current = signal[i]
            if signal[i-1] == 1 and signal[i+1] == 1:
                if signal[i] == 0:
                    raise InterruptedError

            if prev == 0 and current == 1:
                index.append(i)
            prev = current

        return index

    def _get_channel(self, channel_name):
        index = self.labels.index(channel_name)
        return self.channels[index]

    def _get_label(self, sensorName):
        index = self.channels.index(sensorName)
        return self.labels[index]

    def _process_data(self, data, header):
        columns = header['column']
        label = header['label']
        data_dict = {name:[] for name in label}
        data_dict['timestamps'] = []
        print(data_dict)
        for row in data:
            row = row.strip().split("\t")
            for col in columns:
                index = columns.index(col)
                if col in label:
                    data_dict[col].append(int(row[index]))
                elif col == "nSeq":
                    data_dict['timestamps'].append(int(row[index]))
        for key in data_dict.keys():
            data_dict[key] = np.array(data_dict[key])
        return data_dict

    def _process_sensor_file(self, path):
        with open(path, "r") as f:
            header = []
            data = []
            for line in f.readlines():
                if line.startswith("#"):
                    header.append(line.strip())
                else:
                    data.append(line.strip())

            header = json.loads(header[1].strip("#"))
            header = header[list(header.keys())[0]]
        return header, data

@click.command()
@click.option('-v','--videopath', 'videoPath', help='path to the video file')
@click.option('-s', '--sensorpath', 'sensorPath', help='path to the sensor file')
@click.option('-p', '--savePath', 'savePath', help='path to save the created data')
@click.option('--shift/--no-shift', default=False, help='Add the flag to process the setting where the dark settings precede the bright setting')
@click.option('--trigger/--no-trigger', default=False, help='Plot the raw trigger function')
@click.option('--plot/--no-plot', default=True, help="Plot signals after cropping")
def main(videoPath, sensorPath, shift, trigger, plot):

    #sample videopath and sensorpath files
    # videoPath = "/Users/bhargavacharya/phd_projects/notebooks/data/MVI_5525.MOV"
    # sensorPath = "/Users/bhargavacharya/phd_projects/notebooks/data/william.txt"

    #Due to randomization the script can be adapted to deal with two different settings of the protocol
    if shift:
        order = ['LowHR-Dark','LowHR-Bright',   'HighHR-Bright','HighHR-Dark']
    else:
        order = ['LowHR-Bright', 'LowHR-Dark', 'HighHR-Dark', 'HighHR-Bright']
    click.echo(f'Oder for processing the files is : {order}')
    sensor_data(sensorPath, videoPath, order, plot_trigger=trigger, plot=plot)

if __name__ == '__main__':
    main()