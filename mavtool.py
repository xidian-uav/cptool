import json
import multiprocessing
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from pymavlink import mavutil, mavwp, mavextra
from Cptool.config import toolConfig
import sys, select, os


class Location:
    def __init__(self, x, y=None, timeS=0):
        if y is None:
            self.x = x.x
            self.y = x.y
        else:
            self.x = x
            self.y = y
        self.timeS = timeS
        self.npa = np.array([x, y])

    def __sub__(self, other):
        return Location(self.x - other.x, self.y - other.y)

    def __str__(self):
        return f"X: {self.x} ; Y: {self.y}"

    def sum(self):
        return self.npa.sum()

    @classmethod
    def distance(cls, point1, point2):
        if point1.x == 0 or point2.x == 0:
            return 0
        return mavextra.distance_lat_lon(point1.x, point1.y,
                                         point2.x, point2.y)


class OnlineBinRead(multiprocessing.Process):
    def __init__(self, log_file):
        super().__init__()
        self.flight_log = mavutil.mavlink_connection(self.log_file)


def load_param() -> json:
    """
    load parameter we want to fuzzing
    :return:
    """
    if toolConfig.MODE == 'Ardupilot':
        path = 'Cptool/param_ardu.json'
    elif toolConfig.MODE == 'PX4':
        path = 'Cptool/param_px4.json'
    with open(path, 'r') as f:
        return pd.DataFrame(json.loads(f.read()))


def get_default_values(para_dict):
    return para_dict.loc[['default']]


def select_sub_dict(para_dict, param_choice):
    return para_dict[param_choice]


def read_range_from_dict(para_dict):
    return np.array(para_dict.loc['range'].to_list())


def read_unit_from_dict(para_dict):
    return para_dict.loc['step'].to_numpy()


# Log analysis function
def read_path_specified_file(log_path, exe):
    """
        :param log_path:
        :param exe:
        :return:
        """
    file_list = []
    for filename in os.listdir(log_path):
        if filename.endswith(f'.{exe}'):
            file_list.append(filename)
    file_list.sort()
    return file_list


def rename_bin(log_path, ranges):
    file_list = read_path_specified_file(log_path, 'BIN')
    # ????????????????????????.BIN????????????????????????
    for file, num in zip(file_list, range(ranges[0], ranges[1])):
        name, _ = file.split('.')
        os.rename(f"{log_path}/{file}", f"{log_path}/{str(num).zfill(8)}.BIN")


def min_max_scaler_param(param_value):
    para_dict = load_param()
    participle_param = toolConfig.PARAM
    param_choice_dict = select_sub_dict(para_dict, participle_param)

    param_bounds = read_range_from_dict(param_choice_dict)
    lb = param_bounds[:, 0]
    ub = param_bounds[:, 1]
    param_value = (param_value - lb) / (ub - lb)
    return param_value


def return_min_max_scaler_param(param_value):
    param = load_param()
    param_bounds = read_range_from_dict(param)
    lb = param_bounds[:, 0]
    ub = param_bounds[:, 1]
    param_value = (param_value * (ub - lb)) + lb
    return param_value


def min_max_scaler(trans, values):
    status_value = values[:, :toolConfig.STATUS_LEN]
    param_value = values[:, toolConfig.STATUS_LEN:]

    param_value = min_max_scaler_param(param_value)

    status_value = trans.transform(status_value)

    return np.c_[status_value, param_value]


def return_min_max_scaler(trans, values):
    status_value = values[:, :toolConfig.STATUS_LEN]
    param_value = values[:, toolConfig.STATUS_LEN:]

    param_value = return_min_max_scaler_param(param_value)

    status_value = trans.transform(status_value)

    return np.c_[status_value, param_value]


def pad_configuration_default_value(params_value):
    para_dict = load_param()
    # default values
    all_default_value = para_dict.loc[['default']]
    all_default_value = pd.concat([all_default_value] * params_value.shape[0])
    # replace values
    participle_param = toolConfig.PARAM_PART
    all_default_value[participle_param] = params_value
    return all_default_value.values


def _systematicSampling(dataMat, number):
    length = dataMat.shape[0]
    k = length // number
    out = range(length)
    out_index = out[:length:k]
    return out_index


def draw_att_des_and_ach_repair(pdarray, exec='pdf'):
    index = _systematicSampling(pdarray, 500)
    pdarray = pdarray.iloc[index]
    # 'AccX', 'AccY', 'AccZ',
    plt.rcParams['font.sans-serif']=['SimHei']

    plt.rcParams['axes.unicode_minus'] = False

    repair_line = 332 # real 332 ; thrust 305

    for name in ['Roll', 'Pitch', 'Yaw']:
        x = pdarray[name].to_numpy()
        y = pdarray[f"Des{name}"].to_numpy()

        loss = np.abs(x - y)

        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot()

        ax2 = ax1.twinx()

        ax2.fill_betweenx([0, 10 * np.max(loss)], [0, 0],
                          [repair_line, repair_line], color="tomato", alpha=0.2, label="???????????????")

        ax2.fill_betweenx([0, 10 * np.max(np.abs(x - y))], [repair_line, repair_line],
                          [len(x), len(x)], color="green", alpha=0.2, label="??????????????????")

        mid = np.sqrt(x.max() - x.min())

        ax1.plot(y, '-', label='?????????', linewidth=2)
        ax1.plot(x, '--', label='?????????', linewidth=2)
        ax1.set_xlabel("????????? (0.1 ???)", fontsize=18)
        ax1.set_ylabel(f'{name} (deg)', fontsize=18)
        ax1.annotate('????????????', xy=(repair_line, x.min()+mid*0.5),
                     xytext=(repair_line+pdarray.shape[0] * 0.1, x.min()+mid*0.5),
                     arrowprops=dict(arrowstyle="->", color="r", hatch='*',), fontsize='16')

        ax2.bar(np.arange(len(x)), loss, label='??????', color='tab:cyan')
        ax2.set_ylim([0, 10 * np.max(np.abs(x - y))])
        ax2.set_ylabel('?????? ???deg???', fontsize=18)

        fig.legend(loc='upper center', ncol=3, fontsize='18')
        plt.setp(ax1.get_xticklabels(), fontsize=18)
        plt.setp(ax2.get_yticklabels(), fontsize=18)
        plt.setp(ax1.get_yticklabels(), fontsize=18)

        plt.margins(0, 0)
        plt.gcf().subplots_adjust(bottom=0.12)
        # plt.savefig(f'{os.getcwd()}/fig/{toolConfig.MODE}/{self.in_out}/{cmp_name}/{name.lower()}.{exec}')
        plt.subplots_adjust(left=0.125, bottom=0.132, right=0.88, top=0.79, wspace=0.2, hspace=0.2)
        plt.show()
        # plt.clf()


def draw_att_des_and_ach(pdarray, exec='pdf'):
    index = _systematicSampling(pdarray, 300)
    pdarray = pdarray.iloc[index]
    # 'AccX', 'AccY', 'AccZ',
    for name in ['Roll', 'Pitch', 'Yaw']:
        x = pdarray[name].to_numpy()
        y = pdarray[f"Des{name}"].to_numpy()

        loss = np.abs(x - y)

        fig = plt.figure(figsize=(8, 5))
        ax1 = plt.subplot()

        ax2 = ax1.twinx()

        ax2.fill_betweenx([0, 10 * np.max(np.abs(x - y))], [210, 210],
                          [len(x), len(x)], color="tomato", alpha=0.2, label="Unstable Area")

        ax1.plot(y, '-', label='Achieved', linewidth=2)
        ax1.plot(x, '--', label='Desired', linewidth=2)
        ax1.set_xlabel("Timestamp (0.1 Second)", fontsize=18)
        ax1.set_ylabel(f'{name} (deg)', fontsize=18)

        ax2.bar(np.arange(len(x)), loss, label='Bias', color='tab:cyan')
        ax2.set_ylim([0, 10 * np.max(np.abs(x - y))])
        ax2.set_ylabel('Bias (deg)', fontsize=18)

        fig.legend(loc='upper center', ncol=2, fontsize='18')
        plt.setp(ax1.get_xticklabels(), fontsize=18)
        plt.setp(ax2.get_yticklabels(), fontsize=18)
        plt.setp(ax1.get_yticklabels(), fontsize=18)

        plt.margins(0, 0)
        plt.gcf().subplots_adjust(bottom=0.12)
        # plt.savefig(f'{os.getcwd()}/fig/{toolConfig.MODE}/{self.in_out}/{cmp_name}/{name.lower()}.{exec}')
        plt.show()
        # plt.clf()


def sort_result_detect_repair(result_time, detect_time, repair_time):
    """
    check whether the result is appear after detecting and repairing.
    :param result_time:
    :param detect_time:
    :param repair_time:
    :return:
    """
    if result_time > repair_time:
        return "repair"
    if result_time > detect_time:
        return "detect"
    return "miss"
