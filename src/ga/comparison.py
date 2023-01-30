import os
import sys
import timeit
import ga as Ga
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.spatial.distance import cdist

dir = os.path.join(os.getcwd(), 'src', 'model')
sys.path.append(dir)

import model as Model  # noqa

TOTAL_DEVICE: int = 8

# DISTANCE: np.ndarray = np.random.random((TOTAL_DEVICE, 2))
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
    start_time = timeit.default_timer()

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
        connection_specification.append(
            [('===', '===', '===', '===', '===', '===', '===')])

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

    tm1 = timeit.default_timer() - start_time

    print()
    print("Monte Carlo Simulation")
    print(f"Nodes: {sorted(device_specification[-1][0], reverse=True)}")
    print(f"Txpower: {device_specification[-1][1]}")
    print(f"Execution time: ", tm1, "ms")

    '''
    Let the GA do the transmission power optimization begins
    '''
    start_time = timeit.default_timer()

    nodes: list = device_specification[-1][0]
    txpower: list = device_specification[-1][1]
    x: list = [(node, position, transmission_power) for position,
               (node, transmission_power) in enumerate(zip(nodes, txpower))]
    important_nodes: list = sorted(x, key=lambda x: x[0], reverse=True)

    sz = np.int32((np.sqrt(len(device_specification) & ~1)))

    POPULATION_SIZE: int = sz

    # Generate the initial population by replacing the transmission power of each device with newer values
    population: list = [[(nodes[0], nodes[1], np.random.randint(-20, 8))
                        for nodes in important_nodes] for _ in range(POPULATION_SIZE)]

    GA: Ga = Ga.GA(model=MODEL, max_iteration=sz)

    # Calculate the fitness of the initial population
    fitness: list = [GA.fitness(s) for s in population]

    population, fitness = GA.run(population, fitness)

    # Merge the initial nodes with the best solution from the GA
    merge_solution = [(node, i, fitness[0][:-1][i])
                      for i, (node, position, txpower) in enumerate(important_nodes)]

    nodes, txpower = zip(*[(item[0], item[2]) for item in merge_solution])

    tm2 = timeit.default_timer() - start_time

    print()
    print("Genetic Algorithm")
    print(f"Nodes: {sorted(nodes, reverse=True)}")
    print(f"Txpower: {txpower}")
    print(f"Execution time: ", tm2, "ms")

    print()
    print(f"Iteration: {len(device_specification)}")

    print()
    print("Genetic Algorithm is", (tm1 - tm2) / tm1 *
          100, "% faster than Monte Carlo Simulation")
