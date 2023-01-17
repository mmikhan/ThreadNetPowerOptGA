import os
import csv
import random
import itertools
import numpy as np
import collections.abc

from datetime import datetime
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist

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
TOTAL_DEVICES = 8

# Run the model until the condition is met
model = True

'''
Group the iterator into n-sized chunks
    - iterator: iterator to group
    - n: size of the chunk

    - Example:
        - grouper([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
        - [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    
    - Reference: https://stackoverflow.com/a/71951567/3313563
'''
def grouper(iterator: collections.abc.Iterator, n: int) -> collections.abc.Iterator[list]:
    while chunk := list(itertools.islice(iterator, n)):
        yield chunk

# Generate random distance between the devices
distance = np.random.random((TOTAL_DEVICES, 2))

# Calculate the distance between the devices using the euclidean matrix distance
distance_matrix = cdist(distance, distance)

'''
Calculate the path loss between the devices

Log-distance or Log-normal Shadowing path loss model

    - fc: carrier frequency in [Hz]
    - d: distance between the devices in [m]
    - d0: reference distance in [m]
    - n: path loss exponent
    - sigma: variance / standard deviation of the log-normal shadowing in [dB]
'''
def logdist_or_norm(fc, d, d0, n, sigma):
    # Calculate the wavelength
    lamda = 3e8 / fc

    # Calculate the path loss
    PL = -20 * np.log10(lamda / (4 * np.pi * d)) + 10 * n * np.log10(d / d0) + sigma

    # Add noise if sigma is not None
    if sigma:
        return PL + sigma * np.random.randn(d.size)

    return PL

# Path loss model parameters
fc = 2.4e9      # operation in 2.4 GHz
d0 = 0.25       # good choice for indoor distances (microcells)
sigma = 3       # standard deviation of the log-normal shadowing in [dB] / book suggests 3 dB
Exp = 4         # path loss exponent (book suggests 4) / mid value in the obstructed in building range (Table 1.1)

# Antenna parameters
Gt = 1          # transmit antenna gain in [dBi] / No transmitter antenna gain as provided by Nordic Semiconductor (nRF52840) datasheet
Gr = 1          # receive antenna gain in [dBi] / No receiver antenna gain as provided by Nordic Semiconductor (nRF52840) datasheet

RSSI_THRESHOLD = -100    # RSSI threshold in [dBm]
PENALTY = 1000  # default penalty for violating the constraints

# Total penalized transmission power for violating the constraints
total_penalized_transmission_power = 0

# Store all the individual results
individual_result = []

# Store the final result
output = []

# Reverse the dictionary to get the device type from the device label
label = {v:k for k, v in DEVICES.items()}

def plot_distance_matrix(distance_matrix: np.array) -> None:
    """
    Plots a distance matrix using the matplotlib library.

    Args:
        distance_matrix (np.array): Distance matrix to plot.

    Returns:
        None.
    """
    plt.imshow(distance_matrix, cmap='viridis', interpolation='nearest')

    # Write distance values on the plot for each cell
    for i in range(len(distance_matrix)):
        for j in range(len(distance_matrix)):
            plt.text(j, i, round(distance_matrix[i][j], 2), ha="center", va="center", color="w")

    plt.colorbar()
    plt.title("Euclidean Distance Matrix")
    plt.xlabel("Locations"), plt.ylabel("Locations")
    plt.savefig(os.path.join(os.getcwd(), "dist", "distance_matrix.png"), dpi=300)
    plt.show()

# plot_distance_matrix(distance_matrix)

if __name__ == "__main__":
    while model:
        # Generate total random devices based on the types of devices
        nodes = [random.choice(list(DEVICES.values())) for _ in range(TOTAL_DEVICES)]
        
        # Generate transmit power for each device
        txpower = [random.choice([_ for _ in range(-20, TOTAL_DEVICES+1, 4)]) for _ in range(TOTAL_DEVICES)]
        
        # Zip the nodes and txpower together and enumerate them to get the initial position
        x =[(node, position, transmission_power) for position, (node, transmission_power) in enumerate(zip(nodes, txpower))]
        
        # Sort the nodes on descending order of the node type
        important_nodes = sorted(x, key=lambda x: x[0], reverse=True)

        # Path loss between the devices
        path_loss = []

        # Local satisfaction condition for the mathematical model
        satisfied_condition = True

        # Individual penalized transmission power for violating the constraints
        individual_penalized_transmission_power = 0

        # RSSI between the devices
        RSSI = []

        '''
        Filter out the unallocated devices from the RSSI calculation
        since the unallocated devices are not connected to the network
        thus, it doesn't make sense to calculate the RSSI between them

            - However, it's not ideal to filter out the unallocated devices
            from the list of all devices since it will drop the position of the device

            - So, we skip RSSI calculation for the unallocated devices instead
            during the RSSI calculation process by checking the device type first below

        '''
        # important_nodes = [node for node in important_nodes if not node[0] == DEVICES["Unallocated"]]

        '''
        Method 1: RSSI calculation

            - Triangle network formula to calculate the RSSI between each device
            - Generating 3 chunks of the total devices to calculate the RSSI between the devices
        '''
        # chunks = list(grouper(iter(important_nodes), 3))

        # for chunk in chunks:
        #     for comb in itertools.combinations(chunk, 2):
        #         current, next = comb

        #         current_device, current_position, current_transmission_power = current
        #         next_device, next_position, next_transmission_power = next

        #         '''
        #         Filter out the unallocated devices from the RSSI calculation
        #         since the unallocated devices are not connected to the network
        #         thus, it doesn't make sense to calculate the RSSI between them
        #         '''
        #         if not current_device == DEVICES["Unallocated"] and not next_device == DEVICES["Unallocated"]:
        #             # Calculate the Path loss between the devices
        #             pls = logdist_or_norm(fc, distance_matrix[current_position][next_position], d0, Exp, sigma)
        #             path_loss.append(pls[0])

        #             # Calculate the RSSI between the devices
        #             RSSI_Downlink = current_transmission_power - pls[0] + Gt + Gr
        #             RSSI_Uplink = next_transmission_power - pls[0] + Gt + Gr
        #             RSSI.append((RSSI_Downlink, RSSI_Uplink))

        #             # Print the RSSI between the devices
        #             print(f"RSSI between {label[current_device]} and {label[next_device]}: {RSSI_Downlink} dBm and {RSSI_Uplink} dBm")

        #             # Check if the RSSI between the devices is within the constraints
        #             if not RSSI_Downlink > RSSI_THRESHOLD:
        #                 satisfied_condition = False
        #                 total_penalized_transmission_power += PENALTY
        #                 individual_penalized_transmission_power += PENALTY

        #             if not RSSI_Uplink > RSSI_THRESHOLD:
        #                 satisfied_condition = False
        #                 total_penalized_transmission_power += PENALTY
        #                 individual_penalized_transmission_power += PENALTY

        #             individual_result.append({
        #                 "current_device": label[current_device],
        #                 "next_device": label[next_device],
        #                 "path_loss": pls[0],
        #                 "distance": distance_matrix[current_position][next_position],
        #                 "RSSI_Downlink": RSSI_Downlink,
        #                 "RSSI_Uplink": RSSI_Uplink,
        #             })
        
        '''
        Method 2: RSSI calculation

            - Use combinations to calculate the RSSI between each device
            - The combination will generate a possible list of all the connection between the devices
        '''
        for comb in itertools.combinations(important_nodes, 2):
            current, next = comb
            
            current_device, current_position, current_transmission_power = current
            next_device, next_position, next_transmission_power = next

            '''
            Filter out the unallocated devices from the RSSI calculation
            since the unallocated devices are not connected to the network
            thus, it doesn't make sense to calculate the RSSI between them
            '''
            if not current_device == DEVICES["Unallocated"] and not next_device == DEVICES["Unallocated"]:
                # Calculate the Path loss between the devices
                pls = logdist_or_norm(fc, distance_matrix[current_position][next_position], d0, Exp, sigma)
                path_loss.append(pls[0])

                # Calculate the RSSI between the devices
                RSSI_Downlink = current_transmission_power - pls[0] + Gt + Gr
                RSSI_Uplink = next_transmission_power - pls[0] + Gt + Gr
                RSSI.append((RSSI_Downlink, RSSI_Uplink))

                # Print the RSSI between the devices
                # print(f"RSSI Downlink: {RSSI_Downlink} dBm")
                # print(f"RSSI Uplink: {RSSI_Uplink} dBm")

                # Check if the RSSI is greater than -100 dBm
                if not RSSI_Downlink > RSSI_THRESHOLD:
                    satisfied_condition = False
                    total_penalized_transmission_power += PENALTY
                    individual_penalized_transmission_power += PENALTY

                if not RSSI_Uplink > RSSI_THRESHOLD:
                    satisfied_condition = False
                    total_penalized_transmission_power += PENALTY
                    individual_penalized_transmission_power += PENALTY

                individual_result.append({
                    "current_device": current_device,
                    "next_device": next_device,
                    "path_loss": pls[0],
                    "distance": distance_matrix[current_position][next_position],
                    "RSSI_Downlink": RSSI_Downlink,
                    "RSSI_Uplink": RSSI_Uplink,
                })

        '''
        Constraints for the mathematical model.
        The constraints are based on the
        book "Wireless Sensor Networks: A Comprehensive Approach"
        by A. Ghosh and A. K. Roy (2012)
        '''
        # N_REED = N_ROUTER + N_LEADER
        if not nodes.count(DEVICES["REED"]) == nodes.count(DEVICES["Router"]) + nodes.count(DEVICES["Leader"]):
            satisfied_condition = False
            total_penalized_transmission_power += PENALTY
            individual_penalized_transmission_power += PENALTY

        # N_LEADER = 1
        if not nodes.count(DEVICES["Leader"]) == 1:
            satisfied_condition = False
            total_penalized_transmission_power += PENALTY
            individual_penalized_transmission_power += PENALTY

        # N_ROUTER + N_LEADER >= 3
        if not nodes.count(DEVICES["Router"]) + nodes.count(DEVICES["Leader"]) >= 3:
            satisfied_condition = False
            total_penalized_transmission_power += PENALTY
            individual_penalized_transmission_power += PENALTY

        # N_BORDER_ROUTER = 2
        if not nodes.count(DEVICES["Border Router"]) == 2:
            satisfied_condition = False
            total_penalized_transmission_power += PENALTY
            individual_penalized_transmission_power += PENALTY

        if not satisfied_condition:
            '''
            Add the sum of the penalized transmission power
            to the total penalized transmission power
            and the individual penalized transmission power
            '''
            total_penalized_transmission_power += sum(txpower)
            individual_penalized_transmission_power += sum(txpower)

            output.append({
                "type": "Unsatisfied",
                "devices": nodes,
                "labels": [label[n] + str(i+1) for i, n in enumerate(nodes)],
                "txpower": txpower,
                "path_loss": path_loss,
                "RSSI": RSSI,
                "penalty": individual_penalized_transmission_power,
                "total_penalty": total_penalized_transmission_power,
            })
            
            # Add a new line to separate the results
            individual_result.append({})

        if satisfied_condition:
            model = False

            '''
            Reset the total penalized transmission power
            and the individual penalized transmission power
            when the model is satisfied
            '''
            total_penalized_transmission_power = 0
            individual_penalized_transmission_power = 0

            output.append({
                "type": "Satisfied",
                "devices": nodes,
                "labels": [label[n] + str(i+1) for i, n in enumerate(nodes)],
                "txpower": txpower,
                "path_loss": path_loss,
                "RSSI": RSSI,
                "penalty": individual_penalized_transmission_power,
                "total_penalty": total_penalized_transmission_power,
            })

            # GA starts here
            POPULATION_SIZE = 20
            MAX_ITERATION = 100
            # MUTATION_RATE = 0.1
            MUTATION_RATE = 0.6

            # Append the new transmission power to the existing important nodes list
            population = [[tuple(list(i) + [random.randint(-20, 8)]) for i in important_nodes] for _ in range(POPULATION_SIZE)]
            # population = [[tuple(list(i) + [x]) for i in important_nodes for x in [random.choice([_ for _ in range(-20, TOTAL_DEVICES+1, 4)])]] for _ in range(POPULATION_SIZE)]

            def fitness(solution):
                # for i in range(8):
                    # d = distance_matrix[i][i+1]
                    # d = distance_matrix[i][(i+1)%8]
                    # print(d)

                RSSI_Threadhold_penalty = 0
                sc = True

                for comb in itertools.combinations(solution, 2):
                    current, next = comb

                    current_device, current_position, current_transmission_power, current_new_transmission_power = current
                    next_device, next_position, next_transmission_power, next_new_transmission_power = next

                    # Additional for GitHub copilot check
                    # if len(current) == 4 or len(next) == 4:
                    #     current_device, current_position, current_transmission_power, current_new_transmission_power = current
                    #     next_device, next_position, next_transmission_power, next_new_transmission_power = next
                    # else:
                    #     current_device, current_position, current_transmission_power, _, current_new_transmission_power = current
                    #     next_device, next_position, next_transmission_power, _, next_new_transmission_power = next

                    '''
                    Filter out the unallocated devices from the RSSI calculation
                    since the unallocated devices are not connected to the network
                    thus, it doesn't make sense to calculate the RSSI between them
                    '''
                    if not current_device == DEVICES["Unallocated"] and not next_device == DEVICES["Unallocated"]:
                        # Calculate the Path loss between the devices
                        plz = logdist_or_norm(fc, distance_matrix[current_position][next_position], d0, Exp, sigma)

                        # Calculate the RSSI between the devices
                        RSSI_Down = current_new_transmission_power - plz[0] + Gt + Gr
                        RSSI_Up = next_new_transmission_power - plz[0] + Gt + Gr

                        # Check if the RSSI is greater than -100 dBm
                        if not RSSI_Down > RSSI_THRESHOLD:
                            RSSI_Threadhold_penalty += 1000

                        if not RSSI_Up > RSSI_THRESHOLD:
                            RSSI_Threadhold_penalty += 1000

                return [i[3] for i in solution] + [RSSI_Threadhold_penalty]

            fitness_values = [fitness(s) for s in population]

            def selection(population, fitness_values):
                total_fitness = sum(fitness_values)
                relative_fitness = [f/total_fitness for f in fitness_values]
                probabilities = [sum(relative_fitness[:i+1]) for i in range(len(relative_fitness))]

                selected_solution = []
                for i in range(len(population)):
                    r = random.random()
                    for (j, individual) in enumerate(probabilities):
                        if r <= individual:
                            selected_solution.append(population[j])
                            break

                return selected_solution

            def crossover(parent1, parent2):
                # GitHub copilot magic
                crossover_point = random.randint(0, TOTAL_DEVICES-1)

                # crossover_point = random.randint(1, len(parent1) -1)

                child1 = parent1[:crossover_point] + parent2[crossover_point:]
                child2 = parent2[:crossover_point] + parent1[crossover_point:]

                return child1, child2

            def mutation(solution, mutation_rate):
                # for i in range(len(solution)):
                    # GitHub copilot magic (another one)
                    # if(random.random() < mutation_rate):
                    #     solution[i] = (solution[i][0], solution[i][1], solution[i][2], random.randint(-20, 8))

                # GitHub copilot magic (best one)
                for swapped in range(len(solution)):
                    if(random.random() < mutation_rate):
                        swap_with = int(random.random() * len(solution))

                        device, position, transmission_power, new_transmission_power = solution[swapped]
                        device2, position2, transmission_power2, new_transmission_power2 = solution[swap_with]

                        solution[swapped] = (device2, position2, transmission_power2, new_transmission_power2)
                        solution[swap_with] = (device, position, transmission_power, new_transmission_power)

                return solution

            for i in range(MAX_ITERATION):
                selected_solutions = selection(population, [sum(x) for x in fitness_values])

                new_solutions = []
                
                for j in range(0, POPULATION_SIZE, 2):
                    child1, child2 = crossover(selected_solutions[j], selected_solutions[j+1])

                    child1 = mutation(child1, MUTATION_RATE)
                    child2 = mutation(child2, MUTATION_RATE)

                    new_solutions.append(child1)
                    new_solutions.append(child2)

                new_fitness_values = [fitness(s) for s in new_solutions]

                # GitHub copilot magic code for sorting
                # Sort the new population based on the fitness value
                population = [x for _, x in sorted(zip(new_fitness_values, new_solutions))]
                fitness_values = [x for x, _ in sorted(zip(new_fitness_values, new_solutions))]

                print(f"Generation {i+1}: {population[0]}")

            print(f"Best solution: {population[0]}")
            print(f"Best fitness value: {fitness_values[0]}")

            # Plot fitness values over generations
            plt.plot(sorted([sum(x) for x in fitness_values], reverse=True))
            plt.xlabel("Generation")
            plt.ylabel("Fitness Value")
            plt.savefig(os.path.join(os.getcwd(), "dist", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png"))
            plt.show()
            
            # GitHub copilot check
            # new_population = []

            # for i in range(MAX_ITERATION):
            #     # Select the best 50% of the population
            #     population = [x for _, x in sorted(zip(fitness_values, population), key=lambda pair: pair[0], reverse=True)]
            #     population = population[:int(POPULATION_SIZE/2)]

            #     # print(f"Population {population}")

            #     # Generate the next generation
            #     for _ in range(int(POPULATION_SIZE/2)):
            #         # Select two parents
            #         parent1 = random.choice(population)
            #         parent2 = random.choice(population)

            #         # Select a random crossover point
            #         crossover_point = random.randint(0, TOTAL_DEVICES-1)

            #         # Perform crossover
            #         child1 = parent1[:crossover_point] + parent2[crossover_point:]
            #         child2 = parent2[:crossover_point] + parent1[crossover_point:]

            #         # Perform mutation
            #         child1 = [tuple(list(i) + [random.randint(-20, 8)]) for i in child1]
            #         child2 = [tuple(list(i) + [random.randint(-20, 8)]) for i in child2]

            #         # Add the children to the population
            #         # population.append(child1)
            #         # population.append(child2)
            #         new_population.append(child1)
            #         new_population.append(child2)

            #     # Calculate the fitness values of the new population
            #     # fitness_values = [fitness(s) for s in population]
            #     fitness_values = [fitness(s) for s in new_population]

            # # print(f"Population {population}")
            # print(f"Population {new_population}")
            # print(f"Fitness {fitness_values}")

            # # sum fitness values
            # f = [sum(i) for i in fitness_values]

            # # plot fitness sum vs iteration number graph
            # plt.plot(f)
            # plt.xlabel("Iteration")
            # plt.ylabel("Fitness")
            # plt.title("Fitness vs Iteration")
            # plt.savefig(os.path.join(os.getcwd(), "dist/fitness.png"))
            # plt.show()
            # GitHub copilot check end



    # Save the results to a CSV file
    # with open(os.path.join(os.getcwd(), "dist/total-ga.csv"), "w", newline="") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=output[0].keys())
    #     writer.writeheader()
    #     writer.writerows(output)

    # with open(os.path.join(os.getcwd(), "dist/individual-ga.csv"), "w", newline="") as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=individual_result[0].keys())
    #     writer.writeheader()
    #     writer.writerows(individual_result)

    '''
    Plot a semilogx graph of the distances between the devices
    and the path loss between the devices in dB
    '''
    # Flatten the distance matrix to output the distances in one line
    flat_distance = distance_matrix.flatten()

    # Get the unique distances from the distance matrix to plot the graph properly in semilogx graph in dB scale (10log10) sinle straight line
    unique_distance = np.unique(flat_distance)

    # Calculate the path loss between the devices
    y_lognorm = logdist_or_norm(fc, unique_distance, d0, Exp, sigma)

    # # Replace the 0 dB path loss with 0.1 dB to plot the graph properly
    # y_lognorm[y_lognorm == 0] = 0.1
    # y_lognorm[y_lognorm == -0] = -0.1


    # # Replace the -inf dB path loss with 0 dB to plot the graph properly
    # if y_lognorm[0] == -np.inf or y_lognorm[0] == np.inf:
    #     y_lognorm = 0
    
    # # Or
    # if y_lognorm[0] == float("-inf") or y_lognorm[0] == float("inf"):
    #     y_lognorm = 0

    # plt.semilogx(unique_distance, y_lognorm, "k-o")
    # plt.xlabel("Distance (m)"), plt.ylabel("Path Loss (dB)"), plt.grid(True)
    # plt.suptitle("Log-normal Path Loss Model", fontsize=16)
    # plt.title(f"f_c = {fc/1e6}MHz, sigma = {sigma}dB, n = 2", fontsize=10)
    # plt.savefig(os.path.join(os.getcwd(), "dist", "semilogx-ga.png"), dpi=300)
    # # plt.savefig(os.path.join(os.getcwd(), "dist", "semilogx-method1-ga.png"), dpi=300)
    # plt.show()
