import copy
import numpy as np

from queue import Queue, Empty

import carla

class SensorConfigurationInvalid(Exception):
    """
    Exceptions thrown when the sensors used by the agent are not allowed for that specific submissions
    """

    def __init__(self, message):
        super(SensorConfigurationInvalid, self).__init__(message)


class SensorReceivedNoData(Exception):
    """
    Exceptions thrown when the sensors used by the agent take too long to receive data
    """

    def __init__(self, message):
        super(SensorReceivedNoData, self).__init__(message)


class CallBack(object):
    def __init__(self, tag, sensor, data_provider, position):
        self._tag = tag
        self._data_provider = data_provider
        self._position = position

        # TODO: Understand why without saving this, the __call__ function is never called
        self._sensor = sensor

    def __call__(self, data):
        """Parses it into a numpy arrays"""
        if self._position == 'Grayscale':
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = copy.deepcopy(array)
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:,:,0]     # All RGB channels have the same data, so remove 2 of those + alpha channel
            self._data_provider.update_sensor(self._tag, array, data.frame, self._position)
        elif self._position == 'Semantic':
            data.convert(carla.ColorConverter.CityScapesPalette)
            array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
            array = copy.deepcopy(array)
            array = np.reshape(array, (data.height, data.width, 4))
            array = array[:,:,:3]    # Remove the alpha channel
            self._data_provider.update_sensor(self._tag, array, data.frame, self._position)
        else:
            raise ValueError(f"Unexpected value of '{self._position}' for the position")


class SensorInterface(object):
    def __init__(self):
        self._data_buffers = Queue()
        self._queue_timeout = 1

        # TODO. This is hardcoded to cameras at 10 FPS (half the simualtion one),
        # and the initial value is dependent on the amount of ticks during initialization
        self._wait_for_cameras = False

        # Sensors with a 'sensor_tick' don't always send the data in
        # their exact frame, so be a bit lenient manging the queue
        self._num_failed_ticks = 0
        self._max_consecutive_failed_ticks = 5

    def update_sensor(self, tag, data, frame, position):
        """Saves the sensor data into the queue"""
        self._data_buffers.put((tag, frame, data, position))

    def _create_data_dict(self, camera_map):
        """
        Creates the base sensors data. It is either empty or filled up the cameras
        with the ids of the cameras that are active but not sending data on this tick.

        This tells the agent which cameras are active, as well as make it so that we don't wait for their data
        """
        data_dict = {'Grayscale': {}, 'Semantic': {}}
        if not self._wait_for_cameras:
            for camera_id in camera_map['Grayscale'].keys():
                if camera_map['Grayscale'][camera_id][1]:
                    data_dict['Grayscale'][camera_id] = None
            for camera_id in camera_map['Semantic'].keys():
                if camera_map['Semantic'][camera_id][1]:
                    data_dict['Semantic'][camera_id] = None

        return data_dict

    def get_data(self, frame, camera_map):
        """Read the queue to get the sensors data"""
        def wait_for_sensors(sensor_dict, num_sensors):
            return len(sensor_dict['Grayscale'].keys()) + len(sensor_dict['Semantic'].keys()) < num_sensors

        try:
            sensor_dict = self._create_data_dict(camera_map)

            num_sensors = sum([s[1] for s in camera_map['Grayscale'].values()])
            num_sensors += sum([s[1] for s in camera_map['Semantic'].values()])

            while wait_for_sensors(sensor_dict, num_sensors):
                sensor_data = self._data_buffers.get(True, self._queue_timeout)
                if sensor_data[1] != frame:
                    continue

                sensor_dict[sensor_data[3]][sensor_data[0]] = sensor_data[2]

            if self._wait_for_cameras:
                self._num_failed_ticks = 0

        except Empty:
            self._num_failed_ticks += 1
            print("\033[93m'A sensor took too long to send their data\033[0m")
            if self._num_failed_ticks >= self._max_consecutive_failed_ticks:
                raise SensorReceivedNoData("A sensor took too long to send their data")

        self._wait_for_cameras = not self._wait_for_cameras

        return sensor_dict
