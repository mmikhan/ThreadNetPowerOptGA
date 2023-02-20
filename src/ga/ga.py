import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime

dir = os.path.join(os.getcwd(), 'src', 'model')
sys.path.append(dir)

import model as Model  # noqa


class GA:
    def __init__(self, min_txpower: int = -20, max_txpower: int = 8, max_iteration: int = 100, mutation_rate: float = 0.1, model: Model = Model.Model()) -> None:
        '''
        TxPower Min: The minimum transmission power
        '''
        self.MIN_TXPOWER = min_txpower

        '''
        TxPower Max: The maximum transmission power
        '''
        self.MAX_TXPOWER = max_txpower

        '''
        Max Iteration: The maximum number of iterations
        '''
        self.MAX_ITERATION = max_iteration

        '''
        Mutation Rate: The mutation rate
        '''
        self.MUTATION_RATE = mutation_rate

        '''
        Model: The model that will be used to calculate the fitness function
        '''
        self.MODEL: Model = model

    def plot_fitness(self, fitness: list, path: str = os.path.join(os.getcwd(), "dist", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")) -> None:
        '''
        Plot the fitness of the network.

        Args:
            fitness (list): List of fitness values.
            path (str): Path to save the plot.
        '''
        plt.plot(sorted([sum(x) for x in fitness], reverse=True))
        plt.xlabel("Generation"), plt.ylabel("Transmission Power (dBm)")
        plt.suptitle("Transmission Power of the Network", fontsize=16)
        plt.title(
            f"Max: {max([sum(x) for x in fitness])}, Min: {min([sum(x) for x in fitness])}")
        plt.savefig(path, dpi=300)
        plt.close()

    def sphere(self, fitness: list) -> int:
        '''
        Calculate the total transmission power of the network.

        Args:
            power (list): List of transmission power for each device.

        Returns:
            int: Total transmission power of the network.
        '''
        return sum(fitness)

    def fitness(self, solution: list) -> list:
        '''
        Calculate the fitness of the network.

        Args:
            solution (list): List of devices in the network.

        Returns:
            list: List of fitness values.
        '''

        connection: list = self.MODEL.combine_network_topology(
            solution, unallocated=False)

        rssi_penalty: np.ndarray = np.array(
            [connection[i][-1] for i in range(len(connection))]).sum()

        return [i[-1] for i in solution] + [rssi_penalty]

    def selection(self, population: list, fitness: list, method: str = "tournament") -> list:
        '''
        Select the best individuals in the population.

        Args:
            population (list): List of individuals in the population.
            fitness (list): List of fitness values.
            method (str): The selection method. Defaults to "tournament".

        Returns:
            list: List of selected individuals.

        Raises:
            ValueError: The selection method must be either tournament or sorted.
        '''
        if method not in ["tournament", "sorted"]:
            raise ValueError(
                "The selection method must be either tournament or sorted.")

        if method == "sorted":
            # sort population based on fitness values, including penalty
            sorted_population = [
                x for _, x in sorted(zip(fitness, population))]

            selected_population = []

            for i in range(len(population)):
                # select the best individuals in the population
                selected_population.append(sorted_population[i])

            return selected_population

        # Calculate the total fitness of the population
        total_fitness = sum(fitness)

        # Normalize the fitness values
        relative_fitness = [f / total_fitness for f in fitness]

        # Cumulative sum of relative fitness
        probabilities = [sum(relative_fitness[:i + 1])
                         for i in range(len(relative_fitness))]

        # Select the best individuals in the population
        selected_population = []

        for _ in range(len(population)):
            r = random.random()

            for (i, individual) in enumerate(probabilities):
                if r <= individual:
                    selected_population.append(population[i])
                    break

        return selected_population

    def crossover(self, parent1: list, parent2: list) -> tuple:
        '''
        Crossover the best individuals in the population.

        Args:
            parent1 (list): List of devices in the first parent.
            parent2 (list): List of devices in the second parent.

        Returns:
            list: List of devices in the child.
        '''
        # Select a random crossover point
        crossover_point = np.random.randint(0, len(parent1))

        # Create the child
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]

        return child1, child2

    def mutation(self, solution: list, mutation_rate: float, method: str = "swap") -> list:
        '''
        Mutate the child.

        Args:
            solution (list): List of devices in the child.
            mutation_rate (float): The mutation rate.
            method (str): The mutation method. Defaults to "swap".

        Returns:
            list: List of devices in the mutated child.

        Raises:
            ValueError: If the mutation method is not either 'swap' or 'random'.
        '''
        if method not in ["swap", "random"]:
            raise ValueError(
                "The mutation method must be either 'swap' or 'random'")

        if method == "random":
            for i in range(len(solution)):
                if np.random.random() < mutation_rate:
                    solution[i] = (solution[i][0], solution[i][1], random.randint(
                        self.MIN_TXPOWER, self.MAX_TXPOWER))

            return solution

        for swapped in range(len(solution)):
            if np.random.random() < mutation_rate:
                swap_with = int(random.random() * len(solution))

                device, position, transmission_power = solution[swapped]
                device2, position2, transmission_power2 = solution[swap_with]

                solution[swapped] = (device2, position2, transmission_power2)
                solution[swap_with] = (device, position, transmission_power)

        return solution

    def run(self, population: list, fitness: list) -> tuple:
        '''
        Run the genetic algorithm.

        Args:
            population (list): List of individuals in the population.
            fitness (list): List of fitness values.

        Returns:
            tuple: The best individual and its fitness value.

        Raises:
            ValueError: The population and fitness must be specified.
        '''

        if population is None:
            raise ValueError("The population must be specified")

        if fitness is None:
            raise ValueError("The fitness must be specified")

        for i in range(self.MAX_ITERATION):
            selected_population = self.selection(
                population, [sum(x) for x in fitness], method="sorted")

            children = []

            for j in range(0, len(selected_population), 2):
                parent1 = selected_population[j]
                parent2 = selected_population[j+1]

                child1, child2 = self.crossover(parent1, parent2)

                child1 = self.mutation(child1, self.MUTATION_RATE)
                child2 = self.mutation(child2, self.MUTATION_RATE)

                children.append(child1)
                children.append(child2)

            new_fitness = [self.fitness(s) for s in children]

            population = [x for _, x in sorted(
                zip(new_fitness, children), key=lambda pair: pair[0])]
            fitness = [x for x, _ in sorted(
                zip(new_fitness, children), key=lambda pair: pair[0])]

        return population, fitness
