import os
import csv
import itertools
import numpy as np
import collections.abc
import matplotlib.pyplot as plt

from datetime import datetime
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

    def lognorm(self, fc: float = None, distance: np.ndarray | np.float64 = None, d0: float = None, exp: int = None, sigma: int = None, noise: bool = False) -> float | np.ndarray:
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

        if fc is None:
            fc = self.FC

        if distance is None:
            distance = self.DISTANCES

        if d0 is None:
            d0 = self.D0

        if exp is None:
            exp = self.EXP

        if sigma is None:
            sigma = self.SIGMA

        # Calculate the wavelength
        lambda_ = 3e8 / fc

        # Calculate the path loss
        path_loss = 20 * np.log10(lambda_ / (4 * np.pi * distance)) + \
            10 * exp * np.log10(distance / d0) + sigma

        if noise:
            return path_loss + sigma * np.random.randn(distance.size)

        return path_loss

    def calculate_rssi(self, current_transmission_power: int, next_transmission_power: int, path_loss: int) -> tuple[int, int]:

        RSSI_Downlink = current_transmission_power - path_loss + self.GT + self.GR
        RSSI_Uplink = next_transmission_power - path_loss + self.GT + self.GR

        return RSSI_Downlink, RSSI_Uplink

    def rssi_sensitivity_penalty(self, rssi_downlink, rssi_uplink) -> int:
        sensitivity_penalty = 0

        if not rssi_downlink > self.RSSI_THRESHOLD:
            sensitivity_penalty += self.PENALTY

        if not rssi_uplink > self.RSSI_THRESHOLD:
            sensitivity_penalty += self.PENALTY

        return sensitivity_penalty

    def calculate_path_loss(self, current_position: int, next_position: int, noise: bool = False) -> np.float64:
        if self.DISTANCES[current_position][next_position] == 0:
            return np.float64(0)

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

            path_loss = self.calculate_path_loss(
                current_position, next_position)

            RSSI_Downlink, RSSI_Uplink = self.calculate_rssi(
                current_transmission_power, next_transmission_power, path_loss)

            sensitivity_penalty = self.rssi_sensitivity_penalty(
                RSSI_Downlink, RSSI_Uplink)

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

    def mathematical_constraints_penalty(self, nodes: list[int]) -> tuple:
        constraints_penalty = 0

        # N_REED = N_ROUTER + N_LEADER
        if not nodes.count(self.DEVICES["REED"]) == nodes.count(self.DEVICES["Router"]) + nodes.count(self.DEVICES["Leader"]):
            constraints_penalty += self.PENALTY

        # N_LEADER = 1
        if not nodes.count(self.DEVICES["Leader"]) == 1:
            constraints_penalty += self.PENALTY

        # N_ROUTER + N_LEADER >= 3
        if not nodes.count(self.DEVICES["Router"]) + nodes.count(self.DEVICES["Leader"]) >= 3:
            constraints_penalty += self.PENALTY

        # N_BORDER_ROUTER = 2
        if not nodes.count(self.DEVICES["Border Router"]) == 2:
            constraints_penalty += self.PENALTY

        return nodes, constraints_penalty

    def export_to_csv(self, data: list, labels: list = ['Current Device', 'Next Device', 'Distance', 'Path Loss', 'RSSI Downlink', 'RSSI Uplink', 'Sensitivity Penalty'], loc: str = os.path.join(os.getcwd(), "dist", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")):
        with open(loc, "w", newline="") as csvfile:
            if isinstance(data[0], list):
                writer = csv.writer(csvfile)
                writer.writerow(labels)

            if isinstance(data[0], dict):
                writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
                writer.writeheader()

            writer.writerows(data)

    def plot_path_loss_model(self, path: str = os.path.join(os.getcwd(), "dist", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-path-loss')}.png"), noise: bool = False) -> None:
        unique_flatten_distances = np.unique(self.DISTANCES.flatten())

        plt.semilogx(unique_flatten_distances, np.nan_to_num(
            self.lognorm(distance=unique_flatten_distances, noise=noise)), "k-o")
        plt.xlabel("Distance (m)"), plt.ylabel(
            "Path Loss (dB)"), plt.grid(True)
        plt.suptitle("Log-normal Path Loss Model", fontsize=16)
        plt.title(
            f"f_c = {self.FC/1e6}MHz, sigma = {self.SIGMA}dB, Exp = {self.EXP}", fontsize=10)
        plt.savefig(path, dpi=300)
        plt.close()
