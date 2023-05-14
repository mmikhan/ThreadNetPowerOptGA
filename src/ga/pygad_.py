import pygad
import numpy as np

from os import path
from os import getcwd
from datetime import datetime
from itertools import combinations

# distance between devices
distance = np.array(
    [
        0,
        2,
        4,
        6,
        8,
        10,
        12,
        14,
    ]
)


# network parameters
f_c = 2.4e9
n = 5
d_0 = 0.25
sigma = 3


# rssi parameters
G_t = 0
G_r = 0


def path_loss(d, f_c, n, d_0, sigma):
    lambda_c = 3e8 / f_c

    return (
        20 * np.log10(lambda_c / (4 * np.pi * d)) + 10 * n * np.log10(d / d_0) + sigma
    )


def rssi(PP_tx, NP_tx, G_t, G_r, L_p):
    downlink = PP_tx - L_p + G_t + G_r
    uplink = NP_tx - L_p + G_t + G_r

    penalty = 0

    if not downlink > -100:
        penalty += 1000

    if not uplink > -100:
        penalty += 1000

    return penalty


def fitness_func(ga_instance, solution, solution_idx):
    # add index as device number to each item in solution
    solution = np.array(list(zip(solution, range(len(solution)))))

    penalty = 0

    for combination in combinations(solution, 2):
        PP_tx, NP_tx = combination

        # calculate path loss for each combination
        L_p = path_loss(
            np.linalg.norm(distance[PP_tx[1]] - distance[NP_tx[1]]),
            f_c,
            n,
            d_0,
            sigma,
        )

        # calculate rssi for each combination
        penalty = rssi(PP_tx[0], NP_tx[0], G_t, G_r, L_p)

    if not penalty:
        return np.sum(np.abs(solution[:, 0]))

    # higher the penalty, lower the fitness
    return 1 / (penalty + 1)


ga_instance = pygad.GA(
    num_generations=100,
    num_parents_mating=4,
    fitness_func=fitness_func,
    sol_per_pop=20,
    num_genes=8,
    gene_type=int,
    init_range_low=-20,
    init_range_high=8,
    parent_selection_type="rws",
    keep_parents=1,
    crossover_type="single_point",
    mutation_type="scramble",
    mutation_percent_genes=10,
    mutation_num_genes=1,
)

ga_instance.run()
solution, solution_fitness, _ = ga_instance.best_solution()


if solution_fitness > 0 and solution_fitness < 1:
    print("Warning: There's a penalty")
else:
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print(
        "Fitness value of the best solution = {solution_fitness}".format(
            solution_fitness=np.sum(solution)
        )
    )
ga_instance.plot_fitness(
    title="Generation vs. Fitness",
    save_dir=path.join(
        getcwd(), "dist", f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-pygad')}.png"
    ),
)
