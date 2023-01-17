import random
import itertools
import numpy as np

from scipy.spatial.distance import cdist


class GA:
    def __init__(self, devices: dict = {"Unallocated": 0, "SEED": 1, "REED": 2, "Router": 3, "Leader": 4, "Border Router": 5, }, gt: int = 1, gr: int = 1, rssi_threshold: int = -100, penalty: int = 1000, distances: np.array = cdist(np.random.random((8, 2)), np.random.random((8, 2))), total_devices: int = 8, fc: float = 2.4e9, d0: float = 0.25, sigma: int = 3, exp: int = 4, txpower_min: int = -20, txpower_max: int = 8) -> None:
        '''
        Devices: Dictionary of devices
        '''
        self.DEVICES = devices

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

        '''
        Distances: The distance between each device in the network
        '''
        self.DISTANCES = distances

        '''
        Total Devices: The total number of devices in the network
        '''
        self.TOTAL_DEVICES = total_devices

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
        TxPower Min: The minimum transmission power
        '''
        self.TXPOWER_MIN = txpower_min

        '''
        TxPower Max: The maximum transmission power
        '''
        self.TXPOWER_MAX = txpower_max

    @property
    def distances(self) -> np.array:
        '''
        Calculate the distance between each device in the network.

        Returns:
            np.array: Distance matrix.
        '''
        return self.DISTANCES

    @distances.setter
    def distances(self, matrix: np.array) -> None:
        '''
        Set the distance matrix.

        Args:
            matrix (np.array): Distance matrix.
        '''
        if not isinstance(matrix, np.ndarray):
            raise TypeError("The distance matrix must be a numpy array")

        if not matrix.size:
            random_matrix = np.random.random((self.TOTAL_DEVICES, 2))

            self.DISTANCES = random_matrix
        else:
            self.DISTANCES = matrix

    def lognorm(self, fc: float, d: float, d0: float, exp: int, sigma: int, noise: bool = False) -> list:
        '''
        Calculate the path loss between two devices.

        Args:
            fc (float): The carrier frequency.
            d (float): The distance between the two devices.
            d0 (float): The reference distance.
            exp (int): The path loss exponent.
            sigma (int): The standard deviation of the log-normal shadowing model or variance.

        Returns:
            list: The path loss and the standard deviation of the log-normal shadowing model.
        '''

        # Calculate the wavelength
        lambda_ = 3e8 / fc

        # Calculate the path loss
        path_loss = 20 * np.log10(lambda_ / (4 * np.pi * d)) + \
            10 * exp * np.log10(d / d0) + sigma

        if noise:
            return path_loss + sigma * np.random.randn(d.size)

        return [path_loss]

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

        sensitivity_penalty = 0

        for comb in itertools.combinations(solution, 2):
            current, next = comb

            current_device, current_position, current_transmission_power, current_new_transmission_power = current
            next_device, next_position, next_transmission_power, next_new_transmission_power = next

            '''
            Filter out the unallocated devices from the RSSI calculation
            since the unallocated devices are not connected to the network
            thus, it doesn't make sense to calculate the RSSI between them
            '''
            if not current_device == self.DEVICES["Unallocated"] and not next_device == self.DEVICES["Unallocated"]:
                # Calculate the Path loss between the devices
                path_loss = self.lognorm(
                    self.FC, self.DISTANCES[current_position][next_position], self.D0, self.EXP, self.SIGMA)

                # Calculate the RSSI between the devices
                RSSI_Downlink = current_new_transmission_power - \
                    path_loss[0] + self.GT + self.GR
                RSSI_Uplink = next_new_transmission_power - \
                    path_loss[0] + self.GT + self.GR

                # Check if the RSSI is greater than -100 dBm
                if not RSSI_Downlink > self.RSSI_THRESHOLD and not np.isnan(RSSI_Downlink):
                    sensitivity_penalty += self.PENALTY

                if not RSSI_Uplink > self.RSSI_THRESHOLD and not np.isnan(RSSI_Uplink):
                    sensitivity_penalty += self.PENALTY

        return [i[3] for i in solution] + [sensitivity_penalty]

    def selection(self, population: list, fitness: list, method: str = "probability") -> list:
        '''
        Select the best individuals in the population.

        Args:
            population (list): List of individuals in the population.
            fitness (list): List of fitness values.
            method (str): The selection method. Defaults to "probability".

        Returns:
            list: List of selected individuals.

        Raises:
            ValueError: The selection method must be either probability or sort.
        '''

        if method not in ["probability", "sort"]:
            raise ValueError(
                "The selection method must be either probability or sort")

        if method == "sort":
            # Sort the individuals in the population based on their fitness values
            sorted_population = [
                x for _, x in sorted(zip(fitness, population))]

            # Select the best individuals in the population
            selected_population = sorted_population[:len(
                sorted_population) // 2]

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

    def crossover(self, parent1: list, parent2: list) -> list:
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
                    solution[i] = (solution[i][0], solution[i][1], solution[i][2], random.randint(
                        self.TXPOWER_MIN, self.TXPOWER_MAX))

            return solution

        for swapped in range(len(solution)):
            if np.random.random() < mutation_rate:
                swap_with = int(random.random() * len(solution))

                device, position, transmission_power, new_transmission_power = solution[swapped]
                device2, position2, transmission_power2, new_transmission_power2 = solution[
                    swap_with]

                solution[swapped] = (
                    device2, position2, transmission_power2, new_transmission_power2)
                solution[swap_with] = (
                    device, position, transmission_power, new_transmission_power)

        return solution
