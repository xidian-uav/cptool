import logging
import glob
import logging
import multiprocessing
import os
import random
import time
from abc import abstractmethod

import numpy as np
import pandas as pd
import ray
from pymavlink import mavutil, mavwp
from pymavlink.DFReader import DFMessage
from pymavlink.mavutil import mavserial
from pyulog import ULog
from tqdm import tqdm

from Cptool.config import toolConfig
from Cptool.mavtool import load_param, select_sub_dict, read_path_specified_file, sort_result_detect_repair
from ModelFit.approximate import Modeling, CyTCN
from optimize.optimizer import SwarmOptimizer


class DroneMavlink(multiprocessing.Process):
    def __init__(self, port, recv_msg_queue=None, send_msg_queue=None):
        super(DroneMavlink, self).__init__()
        self.recv_msg_queue = recv_msg_queue
        self.send_msg_queue = send_msg_queue
        self._master: mavserial = None
        self._port = port
        self.takeoff = False

    # Mavlink common operation

    def connect(self):
        """
        Connect drone
        :return:
        """
        self._master = mavutil.mavlink_connection('udp:0.0.0.0:{}'.format(self._port))
        try:
            self._master.wait_heartbeat(timeout=30)
        except TimeoutError:
            return False
        logging.info("Heartbeat from system (system %u component %u) from %u" % (
            self._master.target_system, self._master.target_component, self._port))
        return True

    def ready2fly(self) -> bool:
        """
        wait for IMU can work
        :return:
        """
        try:
            while True:
                message = self._master.recv_match(type=['STATUSTEXT'], blocking=True, timeout=30)
                message = message.to_dict()["text"]
                # print(message)
                if toolConfig.MODE == "Ardupilot" and "IMU0 is using GPS" in message:
                    logging.debug("Ready to fly.")
                    return True
                # print(message)
                if toolConfig.MODE == "PX4" and "home set" in message:
                    logging.debug("Ready to fly.")
                    return True
        except Exception as e:
            logging.debug(f"Error {e}")
            return False

    def set_mission(self, mission_file, israndom: bool = False, timeout=30) -> bool:
        """
        Set mission
        :param israndom: random mission order
        :param mission_file: mission file
        :param timeout:
        :return: success
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')

        loader = mavwp.MAVWPLoader()
        loader.target_system = self._master.target_system
        loader.target_component = self._master.target_component
        loader.load(mission_file)
        logging.debug(f"Load mission file {mission_file}")

        # if px4, set home at first
        if toolConfig.MODE == "PX4":
            self.px4_set_home()

        if israndom:
            loader = self.random_mission(loader)
        # clear the waypoint
        self._master.waypoint_clear_all_send()
        # send the waypoint count
        self._master.waypoint_count_send(loader.count())
        seq_list = [True] * loader.count()
        try:
            # looping to send each waypoint information
            # Ardupilot method
            while True in seq_list:
                msg = self._master.recv_match(type=['MISSION_REQUEST'], blocking=True)
                if msg is not None and seq_list[msg.seq] is True:
                    self._master.mav.send(loader.wp(msg.seq))
                    seq_list[msg.seq] = False
                    logging.debug(f'Sending waypoint {msg.seq}')
            mission_ack_msg = self._master.recv_match(type=['MISSION_ACK'], blocking=True, timeout=timeout)
            logging.info(f'Upload mission finish.')
        except TimeoutError:
            logging.warning('Upload mission timeout!')
            return False
        return True

    def start_mission(self):
        """
        Arm and start the flight
        :return:
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        # self._master.set_mode_loiter()

        if toolConfig.MODE == "PX4":
            self._master.set_mode_auto()
            self._master.arducopter_arm()
            self._master.set_mode_auto()
        else:
            self._master.arducopter_arm()
            self._master.set_mode_auto()

        logging.info('Arm and start.')

    def set_param(self, param: str, value: float) -> None:
        """
        set a value of specific parameter
        :param param: name of the parameter
        :param value: float value want to set
        """
        if not self._master:
            raise ValueError('Connect at first!')
        self._master.param_set_send(param, value)
        self.get_param(param)

    def set_params(self, params_dict: dict) -> None:
        """
        set multiple parameter
        :param params_dict: a dict consist of {parameter:values}...
        """
        for param, value in params_dict.items():
            self.set_param(param, value)

    def reset_params(self):
        self.set_param("FORMAT_VERSION", 0)

    def get_param(self, param: str) -> float:
        """
        get current value of a parameter.
        :param param: name
        :return: value of parameter
        """
        self._master.param_fetch_one(param)
        while True:
            message = self._master.recv_match(type=['PARAM_VALUE', 'PARM'], blocking=True).to_dict()
            if message['param_id'] == param:
                logging.debug('name: %s\t value: %f' % (message['param_id'], message['param_value']))
                break
        return message['param_value']

    def get_params(self, params: list) -> dict:
        """
        get current value of a parameters.
        :param params:
        :return: value of parameter
        """
        out_dict = {}
        for param in params:
            out_dict[param] = self.get_param(param)
        return out_dict

    def get_msg(self, msg_type, block=False):
        """
        receive the mavlink message
        :param msg_type:
        :param block:
        :return:
        """
        msg = self._master.recv_match(type=msg_type, blocking=block)
        return msg

    def set_mode(self, mode: str):
        """
        Set flight mode
        :param mode: string type of a mode, it will be convert to an int values.
        :return:
        """
        if not self._master:
            logging.warning('Mavlink handler is not connect!')
            raise ValueError('Connect at first!')
        mode_id = self._master.mode_mapping()[mode]

        self._master.mav.set_mode_send(self._master.target_system,
                                       mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                                       mode_id)
        while True:
            message = self._master.recv_match(type='COMMAND_ACK', blocking=True).to_dict()
            if message['command'] == mavutil.mavlink.MAVLINK_MSG_ID_SET_MODE:
                logging.debug(f'Mode: {mode} Set successful')
                break

    # Special operation
    def set_random_param_and_start(self):
        param_configuration = self.create_random_params(toolConfig.PARAM)
        self.set_params(param_configuration)
        # Unlock the uav
        self.start_mission()

    def px4_set_home(self):
        if toolConfig.HOME is None:
            self._master.mav.command_long_send(self._master.target_system, self._master.target_component,
                                               mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                                               1,
                                               0,
                                               0,
                                               0,
                                               0,
                                               -35.362758,
                                               149.165135,
                                               583.730592)
        else:
            self._master.mav.command_long_send(self._master.target_system, self._master.target_component,
                                               mavutil.mavlink.MAV_CMD_DO_SET_HOME,
                                               1,
                                               0,
                                               0,
                                               0,
                                               0,
                                               40.072842,
                                               -105.230575,
                                               0.000000)
        msg = self._master.recv_match(type=['COMMAND_ACK'], blocking=True, timeout=30)
        logging.debug(f"Home set callback: {msg.command}")

    def gcs_msg_request(self):
        # If it requires manually send the gsc packets.
        pass

    def wait_complete(self, remain_fail=False, timeout=60 * 5):
        """
        Wait the flight mission complete
        :param remain_fail:
        :param timeout:
        :return:
        """
        try:
            timeout_start = time.time()
            while time.time() < timeout_start + timeout:
                # PX4 will manually send the heartbeat for GCS
                self.gcs_msg_request()
                message = self._master.recv_match(type=['STATUSTEXT'], blocking=True, timeout=30)
                if message is None:
                    continue
                message = message.to_dict()
                line = message['text']
                if message["severity"] == 6:
                    if "Land" in line:
                        # if successful landed, break the loop and return true
                        logging.info(f"Successful break the loop.")
                        return True
                elif message["severity"] == 2 or message["severity"] == 0:
                    # Appear error, break loop and return false
                    if "PreArm" in line or remain_fail:
                        # "PreArm" failure will not generate log file, so it not need to delete log
                        # remain_fail means keep this log
                        logging.info(f"Get error with {message['text']}")
                        return True
                    return False
        except (TimeoutError, KeyboardInterrupt) as e:
            # Mission point time out, change other params
            logging.warning(f'Wp timeout! or Key bordInterrupt! exit: {e}')
            return False
        return False

    """
    Internal Methods
    """

    @staticmethod
    def _fill_and_process_public(pd_array: pd.DataFrame):
        """
        Public process of "fill_and_process_pd_log" function
        :param pd_array:
        :return:
        """
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        pd_array = pd_array.drop_duplicates(keep='first')
        # merge data in same TimeS
        df_array = pd.DataFrame(columns=pd_array.columns)
        for group, group_item in pd_array.groupby('TimeS'):
            # fillna
            group_item = group_item.fillna(method='ffill')
            group_item = group_item.fillna(method='bfill')
            df_array.loc[len(df_array.index)] = group_item.mean()
        # Drop nan
        df_array = df_array.fillna(method='ffill')
        df_array = df_array.dropna()

        return df_array

    @staticmethod
    def _order_sort(df_array):
        order_name = toolConfig.STATUS_ORDER.copy()
        param_seq = load_param().columns.to_list()
        param_name = df_array.keys().difference(order_name).to_list()
        param_name.sort(key=lambda item: param_seq.index(item))
        # Status value + Parameter name
        order_name.extend(param_name)
        df_array = df_array[order_name]
        return df_array

    """
    Static method
    """

    @staticmethod
    def create_random_params(param_choice):
        para_dict = load_param()

        param_choice_dict = select_sub_dict(para_dict, param_choice)

        out_dict = {}
        for key, param_range in param_choice_dict.items():
            value = round(random.uniform(param_range['range'][0], param_range['range'][1]) / param_range['step']) * \
                    param_range['step']
            out_dict[key] = value
        return out_dict

    @staticmethod
    def random_mission(loader):
        """
        create random order of a mission
        :param loader: waypoint loader
        :return:
        """
        index = random.sample(loader.wpoints[2:loader.count() - 1], loader.count() - 3)
        index = loader.wpoints[0:2] + index
        index.append(loader.wpoints[-1])
        for i, points in enumerate(index):
            points.seq = i
        loader.wpoints = index
        return loader

    @staticmethod
    def extract_log_path(log_path, skip=True, keep_des=False, threat=None):
        """
        extract and convert bin file to csv
        :param keep_des: whether keep desired value of ATT and RATE
        :param skip: whether skip a log if it has been processed.
        :param log_path:
        :param threat: multiple threat
        :return:
        """

        # If px4, the log is ulg, if ardupilot the log is bin
        global collect_mavlink
        if toolConfig.MODE == "PX4":
            collect_mavlink = CollectMavlinkPX4
            bin_type = "ulg"
        else:
            collect_mavlink = CollectMavlinkAPM
            bin_type = "BIN"

        # read file first
        file_list = read_path_specified_file(log_path, bin_type)

        if not os.path.exists(f"{log_path}/csv"):
            os.makedirs(f"{log_path}/csv")

        # multiple
        if threat is not None:
            arrays = np.array_split(file_list, threat)
            threat_manage = []
            ray.init(include_dashboard=True, dashboard_host="127.0.0.1", dashboard_port=8088)

            for array in arrays:
                threat_manage.append(collect_mavlink.extract_log_path_threat.remote(log_path, array, keep_des, skip))
            ray.get(threat_manage)
            ray.shutdown()
        else:
            # 列出文件夹内所有.BIN结尾的文件并排序
            for file in tqdm(file_list):
                name, _ = file.split('.')
                if skip and os.path.exists(f'{log_path}/csv/{name}.csv'):
                    continue
                # extract
                try:
                    csv_data = collect_mavlink.extract_log_file(log_path + f'/{file}', keep_des)
                    csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)
                except Exception as e:
                    logging.warning(f"Error processing {file} : {e}")
                    continue


class CollectMavlinkAPM(DroneMavlink):
    """
    Mainly responsible for initiating the communication link to interact with UAV
    """

    def __init__(self, port, recv_msg_queue, send_msg_queue):
        super(CollectMavlinkAPM, self).__init__(port, recv_msg_queue, send_msg_queue)

    """
    Static Method
    """

    @staticmethod
    def log_extract_apm(msg: DFMessage, keep_des=False):
        """
        parse the msg of mavlink
        :param keep_des: whether keep att and rate desired and achieve value
        :param msg:
        :return:
        """
        out = None
        if msg.get_type() == 'ATT':
            # if len(toolConfig.LOG_MAP):
            if not keep_des:
                out = {
                    'TimeS': msg.TimeUS / 1000000,
                    'Roll': msg.Roll,
                    'Pitch': msg.Pitch,
                    'Yaw': msg.Yaw,
                }
            else:
                out = {
                    'TimeS': msg.TimeUS / 1000000,
                    "DesRoll": msg.DesRoll,
                    'Roll': msg.Roll,
                    'DesPitch': msg.DesPitch,
                    'Pitch': msg.Pitch,
                    'DesYaw': msg.DesYaw,
                    'Yaw': msg.Yaw,
                }
        elif msg.get_type() == 'RATE':
            if not keep_des:
                out = {
                    'TimeS': msg.TimeUS / 1000000,
                    # deg to rad
                    'RateRoll': msg.R,
                    'RatePitch': msg.P,
                    'RateYaw': msg.Y,
                }
            else:
                out = {
                    'TimeS': msg.TimeUS / 1000000,
                    # deg to rad
                    'DesRateRoll': msg.RDes,
                    'RateRoll': msg.R,
                    'DesRatePitch': msg.PDes,
                    'RatePitch': msg.P,
                    'DesRateYaw': msg.YDes,
                    'RateYaw': msg.Y,
                }
        # elif msg.get_type() == 'POS':
        #     out = {
        #         'TimeS': msg.TimeUS / 1000000,
        #         # deglongtitude
        #         'Lat': msg.Lat,
        #         'Lng': msg.Lng,
        #         'Alt': msg.Alt,
        #     }
        elif msg.get_type() == 'IMU':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'AccX': msg.AccX,
                'AccY': msg.AccY,
                'AccZ': msg.AccZ,
                'GyrX': msg.GyrX,
                'GyrY': msg.GyrY,
                'GyrZ': msg.GyrZ,
            }
        elif msg.get_type() == 'VIBE':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                # m/s^2
                'VibeX': msg.VibeX,
                'VibeY': msg.VibeY,
                'VibeZ': msg.VibeZ,
            }
        elif msg.get_type() == 'MAG':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'MagX': msg.MagX,
                'MagY': msg.MagY,
                'MagZ': msg.MagZ,
            }
        elif msg.get_type() == 'PARM':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                msg.Name: msg.Value
            }
        elif msg.get_type() == 'GPS':
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'Lat': msg.Lat,
                'Lng': msg.Lng,
                'Alt': msg.Alt,
            }
        return out

    @classmethod
    def fill_and_process_pd_log(cls, pd_array: pd.DataFrame):
        """
        pre-process the data collected.
        :param pd_array:
        :return:
        """
        # Remain timestamp .1 and drop duplicate
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        df_array = cls._fill_and_process_public(pd_array)
        # Sort
        df_array = cls._order_sort(df_array)
        return df_array

    @staticmethod
    def extract_log_file(log_file, keep_des=False):
        """
        extract log message form a bin file.
        :param keep_des:
        :param log_file:
        :return:
        """
        accept_item = toolConfig.LOG_MAP

        logs = mavutil.mavlink_connection(log_file)
        # init
        out_data = []
        accpet_param = load_param().columns.to_list()

        while True:
            msg = logs.recv_match(type=accept_item)
            if msg is None:
                break
            # Skip if not index 0 sensor
            # SKip is param is not we want
            if (hasattr(msg, "I") and msg.I != 0) or \
                    (hasattr(msg, "IMU") and msg.IMU != 0) or \
                    (msg.get_type() == 'PARM' and msg.Name not in accpet_param):
                continue
            # Otherwise Record
            out_data.append(CollectMavlinkAPM.log_extract_apm(msg, keep_des))
        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array = CollectMavlinkAPM.fill_and_process_pd_log(pd_array)
        return pd_array

    @staticmethod
    def extract_gps_file(log_file):
        """
        extract gps message form a bin file.
        :param log_file:
        :return:
        """

        logs = mavutil.mavlink_connection(log_file)
        # init
        out_data = []

        while True:
            msg = logs.recv_match(type=["GPS"])
            if msg is None:
                break
            out_data.append(CollectMavlinkAPM.log_extract_apm(msg))
        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        # pd_array = pd_array.drop_duplicates(keep='first')
        return pd_array

    @staticmethod
    def extract_log_file_des_and_ach(log_file):
        """
        extract log message form a bin file with att desired and achieved
        :param log_file:
        :return:
        """

        logs = mavutil.mavlink_connection(log_file)
        # init
        out_data = []

        while True:
            msg = logs.recv_match(type=["ATT"])
            if msg is None:
                break
            out = {
                'TimeS': msg.TimeUS / 1000000,
                'Roll': msg.Roll,
                'DesRoll': msg.DesRoll,
                'Pitch': msg.Pitch,
                'DesPitch': msg.DesPitch,
                'Yaw': msg.Yaw,
                'DesYaw': msg.DesYaw
            }
            out_data.append(out)

        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        pd_array = pd_array.drop_duplicates(keep='first')
        return pd_array

    @staticmethod
    @ray.remote
    def extract_log_path_threat(log_path, file_list, keep_des, skip):
        """
        threat method to extract data from log.
        :param log_path:
        :param file_list:
        :param keep_des: whether keep desired value of ATT and RATE
        :param skip:
        :return:
        """
        for file in tqdm(file_list):
            name, _ = file.split('.')
            if skip and os.path.exists(f'{log_path}/csv/{name}.csv'):
                continue
            try:
                csv_data = CollectMavlinkAPM.extract_log_file(log_path + f'/{file}', keep_des)
                csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)
            except Exception as e:
                logging.warning(f"Error processing {file} : {e}")
                continue
        return True

    # Special function
    @staticmethod
    def random_param_value(param_json: dict):
        """
        random create the value
        :param param_json:
        :return:
        """
        out = {}
        for name, item in param_json.items():
            range = item['range']
            step = item['step']
            random_sample = random.randrange(range[0], range[1], step)
            out[name] = random_sample
        return out

    @staticmethod
    def delete_current_log():
        log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"

        # Read last index
        with open(log_index, 'r') as f:
            num = int(f.readline())
        # To string
        num = f'{num}'
        log_file = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/{num.rjust(8, '0')}.BIN"
        # Remove file
        if os.path.exists(log_file):
            os.remove(log_file)
            # Fix last index number
            last_num = f"{int(num) - 1}"
            with open(log_index, 'w') as f:
                f.write(last_num)

    """
    Thread
    """

    def run(self):
        """
        loop check
        :return:
        """

        while True:
            msg = self._master.recv_match(type=['STATUSTEXT'], blocking=False)
            if msg is not None:
                msg = msg.to_dict()
                # print(msg2)
                if msg['severity'] in [0, 2]:
                    # self.send_msg_queue.put('crash')
                    logging.info('ArduCopter detect Crash.')
                    self.send_msg_queue.put('error')
                    break


class CollectMavlinkPX4(DroneMavlink):
    """
    Mainly responsible for initiating the communication link to interact with UAV
    """

    def __init__(self, port, recv_msg_queue, send_msg_queue):
        super(CollectMavlinkPX4, self).__init__(port, recv_msg_queue, send_msg_queue)

    """
    Method
    """

    def gcs_msg_request(self):
        """
        PX4 needs manual send the heartbeat for GCS
        :return:
        """
        self._master.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                                        mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)

    """
    Static Method
    """

    @classmethod
    def fill_and_process_pd_log(cls, pd_array: pd.DataFrame):
        df_array = cls._fill_and_process_public(pd_array)
        return df_array

    @staticmethod
    def extract_log_file(log_file):
        """
        extract log message form a bin file.
        :param log_file:
        :return:
        """

        ulog = ULog(log_file)

        att = pd.DataFrame(ulog.get_dataset('vehicle_attitude_setpoint').data)[["timestamp",
                                                                                "roll_body", "pitch_body", "yaw_body"]]
        rate = pd.DataFrame(ulog.get_dataset('vehicle_rates_setpoint').data)[["timestamp",
                                                                              "roll", "pitch", "yaw"]]
        acc_gyr = pd.DataFrame(ulog.get_dataset('sensor_combined').data)[["timestamp",
                                                                          "gyro_rad[0]", "gyro_rad[1]", "gyro_rad[2]",
                                                                          "accelerometer_m_s2[0]",
                                                                          "accelerometer_m_s2[1]",
                                                                          "accelerometer_m_s2[2]"]]
        mag = pd.DataFrame(ulog.get_dataset('sensor_mag').data)[["timestamp", "x", "y", "z"]]
        vibe = pd.DataFrame(ulog.get_dataset('sensor_accel').data)[["timestamp", "x", "y", "z"]]
        # Param
        param = pd.Series(ulog.initial_parameters)
        param = param[toolConfig.PARAM]
        # select parameters
        for t, name, value in ulog.changed_parameters:
            if name in toolConfig.PARAM:
                param[name] = round(value, 5)

        att.columns = ["TimeS", "Roll", "Pitch", "Yaw"]
        rate.columns = ["TimeS", "RateRoll", "RatePitch", "RateYaw"]
        acc_gyr.columns = ["TimeS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
        mag.columns = ["TimeS", "MagX", "MagY", "MagZ"]
        vibe.columns = ["TimeS", "VibeX", "VibeY", "VibeZ"]
        # Merge values
        pd_array = pd.concat([att, rate, acc_gyr, mag, vibe]).sort_values(by='TimeS')

        # Process
        df_array = CollectMavlinkPX4.fill_and_process_pd_log(pd_array)
        # Add parameters
        param_values = np.tile(param.values, df_array.shape[0]).reshape(df_array.shape[0], -1)
        df_array[toolConfig.PARAM] = param_values

        # Sort
        order_name = toolConfig.STATUS_ORDER.copy()
        param_seq = load_param().columns.to_list()
        param_name = df_array.keys().difference(order_name).to_list()
        param_name.sort(key=lambda item: param_seq.index(item))

        return df_array

    @staticmethod
    @ray.remote
    def extract_log_path_threat(log_path, file_list, skip):
        for file in tqdm(file_list):
            name, _ = file.split('.')
            if skip and os.path.exists(f'{log_path}/csv/{name}.csv'):
                continue
            try:
                csv_data = CollectMavlinkPX4.extract_log_file(log_path + f'/{file}')
                csv_data.to_csv(f'{log_path}/csv/{name}.csv', index=False)
            except Exception as e:
                logging.warning(f"Error processing {file} : {e}")
                continue
        return True

    @classmethod
    def delete_current_log(cls):
        log_path = f"{toolConfig.PX4_LOG_PATH}/*.ulg"

        list_of_files = glob.glob(log_path)  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        # Remove file
        if os.path.exists(latest_file):
            os.remove(latest_file)


class FlyFixMavlink(DroneMavlink):
    def __init__(self, port, recv_msg_queue=None, send_msg_queue=None):
        super(FlyFixMavlink, self).__init__(port, recv_msg_queue, send_msg_queue)
        self.predictor = None
        self.log_file = None
        self.flight_log = None
        self.read_finish = False
        self.param_current = dict()

    def init_predictor(self, model_class, epochs, batch_size):
        self.predictor: Modeling = model_class(epochs, batch_size, toolConfig.DEBUG)
        self.predictor.read_model()

    def repair_configuration(self, status_data):
        logging.info("Start repair with parameter")
        start = time.time()
        optimize = SwarmOptimizer()
        optimize.set_status(status_data)
        optimize.set_predictor(self.predictor)
        optimize.set_bounds()
        new_config = optimize.start_optimize()
        end = time.time()
        logging.info(f"Repair configuration found and cost: {end - start} second")
        self.set_params(new_config)

    """
    Abstract Method
    """

    @abstractmethod
    def init_binary_log_file(self, device_i=None):
        pass

    @abstractmethod
    def init_current_param(self):
        pass


class FlyFixMavlinkAPM(FlyFixMavlink):
    def __init__(self, port, recv_msg_queue=None, send_msg_queue=None):
        super(FlyFixMavlinkAPM, self).__init__(port, recv_msg_queue, send_msg_queue)

    """
    Initialize Methods
    """

    def init_current_param(self):
        # inti param value
        accpet_param = load_param().columns.to_list()
        while len(self.param_current) <= len(accpet_param):
            msg = self.flight_log.recv_match(type=["PARM"], blocking=True)
            if msg.Name in accpet_param:
                self.param_current.update(CollectMavlinkAPM.log_extract_apm(msg))
        self.param_current.pop('TimeS')
        # logging.debug(f"Current parameters: {self.param_current}")

    def init_binary_log_file(self, device_i=None):
        log_index = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/LASTLOG.TXT"
        # Read last index
        with open(log_index, 'r') as f:
            num = int(f.readline()) + 1
            # To string
        num = f'{num}'
        self.log_file = f"{toolConfig.ARDUPILOT_LOG_PATH}/logs/{num.rjust(8, '0')}.BIN"
        logging.info(f"Current log file: {self.log_file}")

    def read_status_patch_bin(self, time_last, time_unit: float):
        time_unit = float(time_unit)
        out_data = []
        accept_item = toolConfig.LOG_MAP.copy()
        accept_item_ex_param = accept_item.copy()
        accept_item_ex_param.remove("PARM")
        accpet_param = load_param().columns.to_list()

        # Walk to the first message
        while True:
            msg = self.flight_log.recv_match(type=accept_item_ex_param)
            if msg is None:
                return None
            elif msg.TimeUS > time_last:
                # Get first message
                first_msg = msg
                first_time = self.get_time_index_bin(first_msg)
                first_msg = CollectMavlinkAPM.log_extract_apm(first_msg)
                logging.debug(f"Current status at {first_time} second.")

                first_msg.update(self.param_current)
                out_data.append(first_msg)
                new_time = first_time
                break

        # Collect data in one time_unit
        while new_time < first_time + time_unit + 0.1:
            msg = self.flight_log.recv_match(type=accept_item)
            if msg is None:
                break
            if msg.get_type() in ['ATT', 'RATE']:
                out_data.append(CollectMavlinkAPM.log_extract_apm(msg))
            elif msg.get_type() in ['IMU', 'MAG'] and msg.I == 0:
                out_data.append(CollectMavlinkAPM.log_extract_apm(msg))
            elif msg.get_type() == 'VIBE' and msg.IMU == 0:
                out_data.append(CollectMavlinkAPM.log_extract_apm(msg))
            elif msg.get_type() == 'PARM' and msg.Name in accpet_param:
                tmp = CollectMavlinkAPM.log_extract_apm(msg)
                tmp.pop("TimeS")
                self.param_current.update(tmp)
                logging.info(f"Parameters changed: {tmp}")
                continue
            new_time = self.get_time_index_bin(msg)
        # To DataFrame
        pd_array = pd.DataFrame(out_data)
        # Switch sequence, fill,  and return
        pd_array = CollectMavlinkAPM.fill_and_process_pd_log(pd_array)
        return pd_array

    def read_status_patch(self, time_unit: float, status):
        time_unit = float(time_unit)
        out_data = []
        first_msg = self._master.recv_match(type=status, blocking=True)
        out_data.append(FlyFixMavlinkAPM.runtime_extract_apm(first_msg))
        first_time = float(FlyFixMavlinkAPM.get_time_index(first_msg))
        new_time = first_time

        # Collect data in one time_unit
        while new_time < (first_time + time_unit):
            new_msg = self._master.recv_match(type=status, blocking=True)
            # Add and process
            new_time = float(FlyFixMavlinkAPM.get_time_index(new_msg))
            out_data.append(FlyFixMavlinkAPM.runtime_extract_apm(new_msg))

        # Read current configuration
        params = self.get_params(toolConfig.PARAM)
        out_data.append(params)
        pd_array = pd.DataFrame(out_data)

        # Remain timestamp .1 and drop duplicate
        pd_array[toolConfig.PARAM] = pd_array[toolConfig.PARAM].fillna(method="bfill")
        self._fill_and_process_public(pd_array)

        pd_array['TimeS'] = pd_array['TimeS'].round(1)
        df_array = pd_array.drop_duplicates(keep='first')
        # Order
        df_array = self._order_sort(df_array)

        return df_array

    """
    Static Methods
    """

    @staticmethod
    def runtime_extract_apm(msg):
        """
        parse the msg of mavlink
        :param msg:
        :return:
        """
        out = None
        if msg.name == 'ATTITUDE':
            if len(toolConfig.LOG_MAP):
                out = {
                    'TimeS': msg.time_boot_ms / 1000,
                    # Rad
                    'Roll': msg.roll,
                    'Pitch': msg.pitch,
                    'Yaw': msg.pitch,
                    'RateRoll': msg.rollspeed,
                    'RatePitch': msg.pitchspeed,
                    'RateYaw': msg.yawspeed,
                }
        elif msg.name == 'RAW_IMU':
            out = {
                'TimeS': msg.time_usec / 1000000,
                # raw
                'AccX': msg.xacc,
                'AccY': msg.yacc,
                'AccZ': msg.zacc,
                'GyrX': msg.xgyro,
                'GyrY': msg.ygyro,
                'GyrZ': msg.zgyro,
                'MagX': msg.xmag,
                'MagY': msg.ymag,
                'MagZ': msg.zmag,
            }
        # elif msg.name == 'GLOBAL_POSITION_INT':
        #     out = {
        #         'TimeS': msg.time_boot_ms / 1000,
        #         # longtitude
        #         'Lat': msg.lat,
        #         'Lng': msg.lon,
        #         'Alt': msg.alt,
        #     }
        elif msg.name == 'VIBRATION':
            out = {
                'TimeS': msg.time_usec / 1000000,
                # levels
                'VibeX': msg.vibration_x,
                'VibeY': msg.vibration_y,
                'VibeZ': msg.vibration_z
            }
        return out

    # @staticmethod
    # def get_param_from_log(self, param):
    #     self.flight_log.param_fetch_one(param)
    #     while True:
    #         message = self.flight_log.recv_match(type=['PARAM_VALUE', 'PARM'], blocking=True).to_dict()
    #         if message['param_id'] == param:
    #             logging.debug('name: %s\t value: %f' % (message['param_id'], message['param_value']))
    #             break
    #     return message['param_value']
    #
    # @staticmethod
    # def get_params_from_log(self, params):
    #     out_dict = {}
    #     for param in params:
    #         out_dict[param] = DroneMavlink.get_param_from_log(param)
    #     return out_dict

    @staticmethod
    def get_time_index(msg):
        """
        As different message have different time unit. It needs to convert to same second unit.
        :return:
        """
        if msg.name in ["ATTITUDE"]:  # "GLOBAL_POSITION_INT"
            return msg.time_boot_ms / 1000
        if msg.name in ["RAW_IMU", "VIBRATION"]:
            return msg.time_usec / 1000000

    @staticmethod
    def get_time_index_bin(msg):
        """
        As different message have different time unit. It needs to convert to same second unit.
        :return:
        """
        return msg.TimeUS / 1000000

    def wait_bin_ready(self):
        while True:
            time.sleep(0.1)
            if os.path.exists(self.log_file):
                break

    def online_bin_monitor(self, pitch_size_s=3):
        time_last = 0
        accept_item = toolConfig.LOG_MAP.copy()
        # Wait for bin file created
        self.wait_bin_ready()
        file = open(self.log_file, 'rb')

        # Flag
        # detected
        detected_time = 0
        # created repair
        repaired_time = 0
        # upload configuration
        REPAIRED = False
        # fix time
        FIX_TIME = 0
        while True:
            # Exist commands
            if not self.recv_msg_queue.empty():
                # receive error result
                manager_msg, manager_msg_timestamp = self.recv_msg_queue.get()
                # judge the system detect, repair this part.
                detect_repair_result = sort_result_detect_repair(manager_msg_timestamp, detected_time, repaired_time)
                return manager_msg, detect_repair_result

            time.sleep(1)
            # Flush write buffer
            file.flush()
            # Load current log file
            self.flight_log = mavutil.mavlink_connection(self.log_file)
            # inti param value
            self.init_current_param()
            try:
                # Read flight status
                status_data = self.read_status_patch_bin(time_last, pitch_size_s)
                # Check landed or read failure
                if status_data is None:
                    time.sleep(0.1)
                    logging.info(f"Successful break the loop.")
                    return "pass", "repair"
                elif status_data is False:
                    logging.debug(f"Reading status failure, try again.")
                    continue
                # status data to feature data
                feature_data = self.predictor.status2feature(status_data)
                # create predicted status of this status patch
                feature_x, feature_y = self.predictor.data_split(feature_data)

                # Predict
                if isinstance(self.predictor, CyTCN):
                    feature_y = feature_y.reshape((feature_y.shape[0], -1))
                    predicted_feature = self.predictor.predict_feature(feature_x)
                    predicted_feature = predicted_feature.reshape((predicted_feature.shape[0], -1))
                else:
                    predicted_feature = self.predictor.predict_feature(feature_x)

                # deviation loss
                patch_array_loss = self.predictor.cal_patch_deviation(predicted_feature, feature_y)
                patch_max_loss = np.max(patch_array_loss)
                logging.info(f"Time {status_data['TimeS'].iloc[0].round(1)} status' segment max loss:"
                             f" {patch_max_loss}")

                # APM threshold 2.3
                if np.average(patch_max_loss) > 2.3 and not REPAIRED:  # and FIX_TIME < 2: # and not REPAIRED:
                    detected_time = time.time()
                    self.repair_configuration(status_data)
                    repaired_time = time.time()
                    REPAIRED = True
                    # FIX_TIME = FIX_TIME + 1

            except Exception as e:
                logging.warning(f"{e}, then continue looping")

            # Drop old message
            while True:
                msg = self.flight_log.recv_match(type=accept_item)
                if msg is None:
                    break
                else:
                    # Update timestamp
                    time_last = msg.TimeUS


class FlyFixMavlinkPX4(FlyFixMavlink):
    def __init__(self, port, recv_msg_queue=None, send_msg_queue=None):
        super(FlyFixMavlinkPX4, self).__init__(port, recv_msg_queue, send_msg_queue)

    def init_current_param(self):
        # inti param value
        param = pd.Series(self.flight_log.initial_parameters)
        # select parameters
        self.param_current = param[toolConfig.PARAM]

    def init_binary_log_file(self, device_i=None):
        if device_i is None:
            log_path = f"{toolConfig.PX4_LOG_PATH}/*.ulg"

            list_of_files = glob.glob(log_path)  # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            self.log_file = latest_file
            logging.info(f"Current log file: {latest_file}")
        else:
            now = time.localtime()
            now_time = time.strftime("%Y-%m-%d", now)
            log_path = f"{toolConfig.PX4_RUN_PATH}/build/px4_sitl_default/instance_{device_i}/log/{now_time}/*.ulg"

            list_of_files = glob.glob(log_path)  # * means all if need specific format then *.csv
            latest_file = max(list_of_files, key=os.path.getctime)
            self.log_file = latest_file
            logging.info(f"Current log file: {latest_file}")

    def read_status_patch_ulg(self, time_last, time_unit: float):
        time_unit = float(time_unit)

        att = pd.DataFrame(self.flight_log.get_dataset('vehicle_attitude_setpoint').data)[["timestamp",
                                                                                           "roll_body", "pitch_body",
                                                                                           "yaw_body"]]
        rate = pd.DataFrame(self.flight_log.get_dataset('vehicle_rates_setpoint').data)[["timestamp",
                                                                                         "roll", "pitch", "yaw"]]
        acc_gyr = pd.DataFrame(self.flight_log.get_dataset('sensor_combined').data)[["timestamp",
                                                                                     "gyro_rad[0]", "gyro_rad[1]",
                                                                                     "gyro_rad[2]",
                                                                                     "accelerometer_m_s2[0]",
                                                                                     "accelerometer_m_s2[1]",
                                                                                     "accelerometer_m_s2[2]"]]
        mag = pd.DataFrame(self.flight_log.get_dataset('sensor_mag').data)[["timestamp", "x", "y", "z"]]
        vibe = pd.DataFrame(self.flight_log.get_dataset('sensor_accel').data)[["timestamp", "x", "y", "z"]]

        att.columns = ["TimeS", "Roll", "Pitch", "Yaw"]
        rate.columns = ["TimeS", "RateRoll", "RatePitch", "RateYaw"]
        acc_gyr.columns = ["TimeS", "GyrX", "GyrY", "GyrZ", "AccX", "AccY", "AccZ"]
        mag.columns = ["TimeS", "MagX", "MagY", "MagZ"]
        vibe.columns = ["TimeS", "VibeX", "VibeY", "VibeZ"]
        # Merge values
        pd_array = pd.concat([att, rate, acc_gyr, mag, vibe]).sort_values(by='TimeS')

        # Process
        df_array = CollectMavlinkPX4.fill_and_process_pd_log(pd_array)

        # Process
        df_array = df_array[df_array["TimeS"] > time_last]

        # Add parameters
        param_values = np.tile(self.param_current.values, df_array.shape[0]).reshape(df_array.shape[0], -1)
        df_array[toolConfig.PARAM] = param_values

        df_array = df_array[df_array["TimeS"] < (time_last + time_unit + 0.1)]

        return df_array

    @staticmethod
    def runtime_extract_px4(msg):
        """
        parse the msg of mavlink
        :param msg:
        :return:
        """
        out = None
        if msg.name == 'ATTITUDE':
            if len(toolConfig.LOG_MAP):
                out = {
                    'TimeS': msg.time_boot_ms / 1000,
                    # Rad
                    'Roll': msg.roll,
                    'Pitch': msg.pitch,
                    'Yaw': msg.pitch,
                    'RateRoll': msg.rollspeed,
                    'RatePitch': msg.pitchspeed,
                    'RateYaw': msg.yawspeed,
                }
        elif msg.name == 'RAW_IMU':
            out = {
                'TimeS': msg.time_usec / 1000000,
                # raw
                'AccX': msg.xacc,
                'AccY': msg.yacc,
                'AccZ': msg.zacc,
                'GyrX': msg.xgyro,
                'GyrY': msg.ygyro,
                'GyrZ': msg.zgyro,
                'MagX': msg.xmag,
                'MagY': msg.ymag,
                'MagZ': msg.zmag,
            }
        # elif msg.name == 'GLOBAL_POSITION_INT':
        #     out = {
        #         'TimeS': msg.time_boot_ms / 1000,
        #         # longtitude
        #         'Lat': msg.lat,
        #         'Lng': msg.lon,
        #         'Alt': msg.alt,
        #     }
        elif msg.name == 'VIBRATION':
            out = {
                'TimeS': msg.time_usec / 1000000,
                # levels
                'VibeX': msg.vibration_x,
                'VibeY': msg.vibration_y,
                'VibeZ': msg.vibration_z
            }
        return out

    def online_ulg_monitor(self, pitch_size_s=3):
        time_last = 0
        file = open(self.log_file, 'rb')

        # Flag
        # detected
        detected_time = 0
        # created repair
        repaired_time = 0
        # upload configuration
        REPAIRED = False

        while True:
            # Exist commands
            if not self.recv_msg_queue.empty():
                # receive error result
                manager_msg, manager_msg_timestamp = self.recv_msg_queue.get()
                # judge the system detect, repair this part.
                detect_repair_result = sort_result_detect_repair(manager_msg_timestamp, detected_time, repaired_time)
                return manager_msg, detect_repair_result

            time.sleep(1)
            # Flush write buffer
            file.flush()
            # Load current log file
            self.flight_log = ULog(self.log_file)
            # inti param value
            self.init_current_param()
            if True:
                # Read flight status
                status_data = self.read_status_patch_ulg(time_last, pitch_size_s)
                # Check landed or read failure
                if status_data is None:
                    time.sleep(0.1)
                    logging.info(f"Successful break the loop.")
                    return True
                elif status_data is False:
                    logging.debug(f"Reading status failure, try again.")
                    continue
                # status data to feature data
                feature_data = self.predictor.status2feature(status_data)
                # create predicted status of this status patch
                feature_x, feature_y = self.predictor.data_split(feature_data)
                if isinstance(self.predictor, CyTCN):
                    feature_y = feature_y.reshape((feature_y.shape[0], -1))
                # Predict
                predicted_feature = self.predictor.predict_feature(feature_x)
                if isinstance(self.predictor, CyTCN):
                    predicted_feature = predicted_feature.reshape((predicted_feature.shape[0], -1))
                # deviation loss
                patch_array_loss = self.predictor.cal_patch_deviation(predicted_feature, feature_y)
                patch_average_loss = np.average(patch_array_loss)
                logging.info(f"Time {status_data['TimeS'].iloc[0].round(1)} status' patch average loss:"
                             f" {patch_average_loss}")

                # PX4 threshold 2.06
                if np.average(patch_average_loss) > 2.06 and not REPAIRED:
                    detected_time = time.time()
                    self.repair_configuration(status_data)
                    repaired_time = time.time()
                    REPAIRED = True

            # except Exception as e:
            #     logging.warning(f"{e}, then continue looping")

            # Drop old message
            msg = self.flight_log.last_timestamp
            time_last = msg / 1000000
