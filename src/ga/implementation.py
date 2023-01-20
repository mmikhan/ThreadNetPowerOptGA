import os
import random
import numpy as np
import matplotlib.pyplot as plt
from ga import GA as genetic_algorithm

from datetime import datetime
from scipy.spatial.distance import cdist

# Initialize the GA class
GA = genetic_algorithm()

# GA Parameters
POPULATION_SIZE = 20
MAX_ITERATION = 100
MUTATION_RATE = 0.1

# Device types
DEVICES = {
    "Unallocated": 0,
    "SEED": 1,
    "REED": 2,
    "Router": 3,
    "Leader": 4,
    "Border Router": 5,
}

# Number of devices in the network
TOTAL_DEVICES = 28

# Generate random distances between each device
distance = np.random.random((TOTAL_DEVICES, 2))
# distance = np.random.randint(0, 100, size=(TOTAL_DEVICES, 2))
GA.DISTANCES = cdist(distance, distance)

# Generate total random devices based on the types of devices
nodes = [random.choice(list(DEVICES.values())) for _ in range(TOTAL_DEVICES)]

# Generate transmit power for each device
txpower = [random.choice([_ for _ in range(-20, TOTAL_DEVICES+1, 4)])
           for _ in range(TOTAL_DEVICES)]

# Zip the nodes and txpower together and enumerate them to get the initial position
x = [(node, position, transmission_power)
     for position, (node, transmission_power) in enumerate(zip(nodes, txpower))]

# Sort the nodes on descending order of the node type
important_nodes = sorted(x, key=lambda x: x[0], reverse=True)

# Generate the initial population
population = [[tuple(list(i) + [random.randint(-20, 8)])
               for i in important_nodes] for _ in range(POPULATION_SIZE)]

# Calculate the fitness of the initial population
fitness = [GA.fitness(s) for s in population]


def main(MAX_ITERATION, MUTATION_RATE, population, fitness) -> tuple:
    for i in range(MAX_ITERATION):
        selected_population = GA.selection(
            population, [sum(x) for x in fitness])

        children = []

        for j in range(0, len(selected_population), 2):
            parent1 = selected_population[j]
            parent2 = selected_population[j+1]

            child1, child2 = GA.crossover(parent1, parent2)

            child1 = GA.mutation(child1, MUTATION_RATE)
            child2 = GA.mutation(child2, MUTATION_RATE)

            children.append(child1)
            children.append(child2)

        new_fitness = [GA.fitness(s) for s in children]

        population = [x for _, x in sorted(
            zip(new_fitness, children), key=lambda pair: pair[0])]
        fitness = [x for x, _ in sorted(
            zip(new_fitness, children), key=lambda pair: pair[0])]

        print("Iteration: {}, Best Fitness: {}".format(i, min(fitness)))

    return population, fitness


if __name__ == "__main__":
    population, fitness = main(
        MAX_ITERATION, MUTATION_RATE, population, fitness)

    print("Lowest transmission power: {} dBm".format(GA.sphere(fitness[0])))
    print("Best solution: {}".format(population[0]))
    print("Fitness: {}".format(fitness[0]))
    print("Difference between the initial and best transmission power: {}".format(
        GA.sphere(txpower) - GA.sphere(fitness[0])))
    print("Difference between the worst and best transmission power: {}".format(
        GA.sphere(fitness[-1]) - GA.sphere(fitness[0])))

    # Plot the fitness values of the population over the generations
    plt.plot(sorted([sum(x) for x in fitness], reverse=True))
    plt.xlabel("Generation")
    plt.ylabel("Fitness Value")
    plt.savefig(os.path.join(os.getcwd(), "dist",
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"))
    plt.show()
