import os
import csv
import random
import itertools
import numpy as np
import collections.abc

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
                print(f"RSSI Downlink: {RSSI_Downlink} dBm")
                print(f"RSSI Uplink: {RSSI_Uplink} dBm")

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

    # Save the results to a CSV file
    with open(os.path.join(os.getcwd(), "dist/total.csv"), "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)

    with open(os.path.join(os.getcwd(), "dist/individual.csv"), "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=individual_result[0].keys())
        writer.writeheader()
        writer.writerows(individual_result)

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

    plt.semilogx(unique_distance, y_lognorm, "k-o")
    plt.xlabel("Distance (m)"), plt.ylabel("Path Loss (dB)"), plt.grid(True)
    plt.suptitle("Log-normal Path Loss Model", fontsize=16)
    plt.title(f"f_c = {fc/1e6}MHz, sigma = {sigma}dB, n = 2", fontsize=10)
    plt.savefig(os.path.join(os.getcwd(), "dist/semilogx.png"), dpi=300)
    # plt.savefig(os.path.join(os.getcwd(), "dist/semilogx-method1.png"), dpi=300)
    plt.show()
