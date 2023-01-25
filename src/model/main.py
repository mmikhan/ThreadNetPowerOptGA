import os
import csv
import tabulate
import numpy as np
import model as Model

from datetime import datetime
from scipy.spatial.distance import cdist

TOTAL_DEVICE: int = 8

DISTANCE: np.ndarray = np.array([[1, 10], [1, 15], [1, 20], [1, 25], [
                                1, 30], [1, 35], [1, 40], [1, 45]])

MODEL: Model = Model.Model(total_device=TOTAL_DEVICE,
                           distances=cdist(DISTANCE, DISTANCE))

network: bool = True

total_penalty: list = []

device_specification: list = []

# network_specification: list = []
connection_specification: list = []

each_penalized_transmission_power: np.ndarray = []

if __name__ == '__main__':
    while network:
        nodes: list = [np.random.choice(
            list(MODEL.DEVICES.values())) for _ in range(TOTAL_DEVICE)]

        txpower: list = [np.random.choice(
            [_ for _ in range(-20, TOTAL_DEVICE+1, 4)]) for _ in range(TOTAL_DEVICE)]

        x: list = [(node, position, transmission_power) for position,
                   (node, transmission_power) in enumerate(zip(nodes, txpower))]

        important_nodes: list = sorted(x, key=lambda x: x[0], reverse=True)

        _, PENALTY = MODEL.mathematical_constraints_penalty(nodes)

        connection: list = MODEL.combine_network_topology(
            important_nodes, unallocated=False)

        connection_specification.append(connection)
        connection_specification.append([()])

        each_penalized_transmission_power.append(
            np.array([[connection[i][6] for i in range(len(connection))] + [PENALTY] + [sum(txpower)]]))

        device_specification.append([nodes, txpower, sum(txpower), [PENALTY], [sum(
            [connection[i][6] for i in range(len(connection))])], [each_penalized_transmission_power[-1].sum()]])

        if PENALTY == 0 and [sum([connection[i][6] for i in range(len(connection))])][0] == 0:
            network = False

    MODEL.export_to_csv(device_specification, ['Nodes', 'Txpower', 'Total Txpower', 'Penalty', 'RSSI Penalty', 'Entire Power'], os.path.join(
        os.getcwd(), "dist", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-d')}.csv"))

    print(tabulate.tabulate(device_specification, headers=[
          'Nodes', 'Txpower', 'Total Txpower', 'Penalty', 'RSSI Penalty', 'Entire Txpower']))

    flatten_data = [[*i] for j in connection_specification for i in j]
    MODEL.export_to_csv(flatten_data, ['Current Device', 'Next Device', 'Distance',
                        'Path Loss', 'RSSI Downlink', 'RSSI Uplink', 'Sensitivity Penalty'])
    print(tabulate.tabulate(flatten_data, headers=[
          'Current Device', 'Next Device', 'Distance', 'Path Loss', 'RSSI Downlink', 'RSSI Uplink', 'Sensitivity Penalty']))
