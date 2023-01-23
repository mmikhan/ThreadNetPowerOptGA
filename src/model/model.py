import sys
import itertools
import numpy as np
import collections.abc
from scipy.spatial.distance import cdist


class Model:
    def __init__(self, devices: dict = {"Unallocated": 0, "SEED": 1, "REED": 2, "Router": 3, "Leader": 4, "Border Router": 5, }, total_device: int = 8, distances: np.array = cdist(np.random.random((8, 2)), np.random.random((8, 2))), fc: float = 2.4e9, d0: float = 0.25, sigma: int = 3, exp: int = 4, gt: int = 1, gr: int = 1, rssi_threshold: int = -100, penalty: int = 1000):
        '''
        Devices: Dictionary of devices
        '''
        self.DEVICES = devices

        '''
        Distances: The distances between the devices
        '''
        self.DISTANCES = distances

        '''
        Total Device: The total number of devices
        '''
        self.TOTAL_DEVICE = total_device

        '''
        fc: The carrier frequency
        '''
        self.FC = fc

        '''
        d0: The reference distance
        '''
        self.D0 = d0

        '''
        sigma: The standard deviation of the log-normal shadowing model or variance
        '''
        self.SIGMA = sigma

        '''
        Exp: The path loss exponent
        '''
        self.EXP = exp

        '''
        Gt: Transmitter antenna gain
        '''
        self.GT = gt

        '''
        Gr: Receiver antenna gain
        '''
        self.GR = gr

        '''
        RSSI Threshold: The minimum RSSI value that a device can receive
        '''
        self.RSSI_THRESHOLD = rssi_threshold

        '''
        Penalty: The penalty that will be added to the fitness function
        if the RSSI value is less than the RSSI Threshold
        '''
        self.PENALTY = penalty

    def lognorm(self, fc: float, distance: np.ndarray | np.float64, d0: float, exp: int, sigma: int, noise: bool = False) -> float | np.ndarray:
        '''
        Calculate the path loss between two devices.

        Args:
            fc (float): The carrier frequency.
            distance (np.ndarray | np.float64): The distance between the two devices.
            d0 (float): The reference distance.
            exp (int): The path loss exponent.
            sigma (int): The standard deviation of the log-normal shadowing model or variance.

        Returns:
            float | np.ndarray: The path loss.
        '''

        # Calculate the wavelength
        lambda_ = 3e8 / fc

        # Calculate the path loss
        path_loss = 20 * np.log10(lambda_ / (4 * np.pi * distance)) + \
            10 * exp * np.log10(distance / d0) + sigma

        if noise:
            return path_loss + sigma * np.random.randn(distance.size)[0]

        return path_loss

    def calculate_rssi(self, current_position: int, next_position: int, current_transmission_power: int, next_transmission_power: int) -> tuple[int, int, int]:

        path_loss = self.calculate_path_loss(current_position, next_position)

        RSSI_Downlink = current_transmission_power - path_loss + self.GT + self.GR
        RSSI_Uplink = next_transmission_power - path_loss + self.GT + self.GR

        sensitivity_penalty = self.rssi_sensitivity_penalty(
            RSSI_Downlink, RSSI_Uplink)

        return RSSI_Downlink, RSSI_Uplink, path_loss, sensitivity_penalty

    def rssi_sensitivity_penalty(self, rssi_downlink, rssi_uplink):
        sensitivity_penalty = 0

        if not rssi_downlink > self.RSSI_THRESHOLD:
            sensitivity_penalty += self.PENALTY

        if not rssi_uplink > self.RSSI_THRESHOLD:
            sensitivity_penalty += self.PENALTY

        return sensitivity_penalty

    def calculate_path_loss(self, current_position, next_position, noise: bool = False):
        if self.DISTANCES[current_position][next_position] == 0:
            return 0

        return self.lognorm(
            self.FC, self.DISTANCES[current_position][next_position], self.D0, self.EXP, self.SIGMA, noise)

    def triangle_combine_network_topology(self, nodes: list[int]) -> tuple[int, int, int]:
        result = []

        chunks = list(self.grouper(iter(nodes), 3))

        for chunk in chunks:
            specification = self.combine_network_topology(
                chunk, unallocated=True)

            result.append(specification)

        return result

    def combine_network_topology(self, nodes: list[int], unallocated: bool = False) -> tuple[int, int, int]:
        result = []

        if not unallocated:
            nodes = [device for device in nodes if not device[0]
                     == self.DEVICES["Unallocated"]]

        for node in itertools.combinations(nodes, 2):
            current, next = node

            current_device, current_position, current_transmission_power = current
            next_device, next_position, next_transmission_power = next

            RSSI_Downlink, RSSI_Uplink, path_loss, sensitivity_penalty = self.calculate_rssi(
                current_position, next_position, current_transmission_power, next_transmission_power)

            result.append((current_device, next_device, self.DISTANCES[current_device]
                          [next_device], path_loss, RSSI_Downlink, RSSI_Uplink, sensitivity_penalty))

        return result

    def grouper(self, iterator: collections.abc.Iterator, n: int) -> collections.abc.Iterator[list]:
        '''
        Group the iterator into n-sized chunks
            - iterator: iterator to group
            - n: size of the chunk

            - Example:
                - grouper([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
                - [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

            - Reference: https://stackoverflow.com/a/71951567/3313563
        '''
        while chunk := list(itertools.islice(iterator, n)):
            yield chunk
