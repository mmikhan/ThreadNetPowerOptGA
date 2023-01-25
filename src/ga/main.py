import os
import sys
import tabulate
import numpy as np

from datetime import datetime
from scipy.spatial.distance import cdist

dir = os.path.join(os.getcwd(), 'src', 'model')
sys.path.append(dir)

import model as Model  # noqa

TOTAL_DEVICE: int = 8

DISTANCE: np.ndarray = np.array([[1, 10], [1, 15], [1, 20], [1, 25], [
                                1, 30], [1, 35], [1, 40], [1, 45]])

MODEL: Model = Model.Model(total_device=TOTAL_DEVICE,
                           distances=cdist(DISTANCE, DISTANCE))

network: bool = True

total_penalty: list = []

device_specification: list = []

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

        # Store the connection specification for each device to export to csv
        connection_specification.append(connection)
        connection_specification.append([()])

        # Sum of the entire transmission power:
        # sum of penalty from each combination of nodes + penalty from device type selection + sum of transmission power
        entire_txpower: np.ndarray = np.array([connection[i][6] for i in range(
            len(connection))] + [PENALTY] + [sum(txpower)]).sum()

        # Sum of the RSSI penalty from each combination of nodes
        rssi_penalty: np.ndarray = np.array(
            [connection[i][6] for i in range(len(connection))]).sum()

        # Store the device specification with transmission powers to export to csv
        device_specification.append([nodes, txpower, sum(
            txpower), PENALTY, rssi_penalty, entire_txpower])

        if PENALTY == 0 and rssi_penalty == 0:
            network = False

    MODEL.export_to_csv(device_specification, ['Nodes', 'Txpower', 'Total Txpower', 'Penalty', 'RSSI Penalty', 'Entire Power'], os.path.join(
        os.getcwd(), "dist", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-d')}.csv"))
    print(tabulate.tabulate(device_specification, headers=[
          'Nodes', 'Txpower', 'Total Txpower', 'Penalty', 'RSSI Penalty', 'Entire Txpower']))

    flatten_connection_spec = [[*i] for j in connection_specification for i in j]
    MODEL.export_to_csv(flatten_connection_spec, ['Current Device', 'Next Device', 'Distance',
                        'Path Loss', 'RSSI Downlink', 'RSSI Uplink', 'Sensitivity Penalty'])
    print(tabulate.tabulate(flatten_connection_spec, headers=[
          'Current Device', 'Next Device', 'Distance', 'Path Loss', 'RSSI Downlink', 'RSSI Uplink', 'Sensitivity Penalty']))

    MODEL.plot_path_loss_model(noise=True)
