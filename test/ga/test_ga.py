import os
import sys
import unittest
import numpy as np

from datetime import datetime

ga_dir = os.path.join(os.getcwd(), 'src', 'ga')
sys.path.append(ga_dir)

model_dir = os.path.join(os.getcwd(), 'src', 'model')
sys.path.append(model_dir)

import ga as Ga  # noqa
import model as Model  # noqa


class TestGA(unittest.TestCase):
    def setUp(self) -> None:
        self.GA = Ga.GA()
        self.MODEL = Model.Model()

        _NODES = [1, 2, 3]
        _POSITION = [2, 3, 4]
        _TXPOWER = [-2, 4, 8]

        _POPULATION_SIZE = 10

        self.POPULATION = [[(node, position, txpower) for (node, position, txpower) in zip(
            _NODES, _POSITION, _TXPOWER)] for _ in range(_POPULATION_SIZE)]

        self.FITNESS: list = [self.GA.fitness(s) for s in self.POPULATION]

        return super().setUp()

    def test_default_min_txpower(self) -> None:
        self.assertEqual(self.GA.MIN_TXPOWER, -20)

    def test_given_min_txpower(self) -> None:
        ga = Ga.GA(min_txpower=-10)

        self.assertEqual(ga.MIN_TXPOWER, -10)

    def test_default_max_txpower(self) -> None:
        self.assertEqual(self.GA.MAX_TXPOWER, 8)

    def test_given_max_txpower(self) -> None:
        ga = Ga.GA(max_txpower=10)

        self.assertEqual(ga.MAX_TXPOWER, 10)

    def test_default_max_iteration(self) -> None:
        self.assertEqual(self.GA.MAX_ITERATION, 100)

    def test_given_max_iteration(self) -> None:
        ga = Ga.GA(max_iteration=200)

        self.assertEqual(ga.MAX_ITERATION, 200)

    def test_default_mutation_rate(self) -> None:
        self.assertEqual(self.GA.MUTATION_RATE, 0.1)

    def test_given_mutation_rate(self) -> None:
        ga = Ga.GA(mutation_rate=0.2)

        self.assertEqual(ga.MUTATION_RATE, 0.2)

    def test_default_model_instance(self) -> None:
        self.assertIsInstance(self.GA.MODEL, Model.Model)

    def test_given_model_instance(self) -> None:
        ga = Ga.GA(model=self.MODEL)

        self.assertIsInstance(ga.MODEL, Model.Model)

    def test_plot_fitness(self) -> None:
        fitness = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        path: str = os.path.join(
            os.getcwd(), 'dist', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png")

        self.GA.plot_fitness(fitness, path)

        self.assertTrue(path)
        self.assertTrue(os.path.exists(path))

        # Remove the file after the test
        os.remove(path)

        self.assertFalse(os.path.exists(path))

    def test_sphere_for_all_fitness(self) -> None:
        self.assertEqual(self.GA.sphere(np.array(self.FITNESS).flatten()), 100)

    def test_sphere_for_lowest_fitness(self) -> None:
        self.assertEqual(self.GA.sphere(self.FITNESS[0]), 10)

    def test_sphere_for_highest_fitness(self) -> None:
        self.assertEqual(self.GA.sphere(self.FITNESS[-1]), 10)

    def test_fitness_return_only_transmission_power_from_population(self) -> None:
        self.assertEqual(self.GA.fitness(self.POPULATION[0])[
                         :-1], [(txpower) for (_, _, txpower) in self.POPULATION[0]])

    def test_fitness_return_include_rssi_penalty(self) -> None:
        self.assertEqual(len(self.GA.fitness(self.POPULATION[0])), 4)

    def test_selection_return_same_length_of_population(self) -> None:
        self.assertEqual(len(self.GA.selection(
            self.POPULATION, self.FITNESS[0])), len(self.POPULATION))

    def test_selection_return_population_cumulatively(self) -> None:
        self.assertNotEqual(self.GA.selection(
            self.POPULATION, self.FITNESS[0]), self.POPULATION[0])

    def test_crossover_return_two_length_of_separate_parents(self) -> None:
        selected_population = self.GA.selection(
            self.POPULATION, self.FITNESS[0])

        parent1, parent2 = selected_population[0], selected_population[1]

        self.assertEqual(len(self.GA.crossover(parent1, parent2)), 2)

    def test_crossover_return_a_tuple(self) -> None:
        selected_population = self.GA.selection(
            self.POPULATION, self.FITNESS[0])

        parent1, parent2 = selected_population[0], selected_population[-1]

        self.assertIsInstance(self.GA.crossover(parent1, parent2), tuple)

    def test_mutation_return_same_or_swapped_child_elements(self) -> None:
        selected_population = self.GA.selection(
            self.POPULATION, self.FITNESS[0])

        parent1, parent2 = selected_population[0], selected_population[-1]

        child1, child2 = self.GA.crossover(parent1, parent2)

        self.assertCountEqual(self.GA.mutation(
            child1, 10.0), child1)
        self.assertCountEqual(self.GA.mutation(
            child2, 10.0), child2)

        self.assertCountEqual(self.GA.mutation(
            child1, self.GA.MUTATION_RATE), child1)
        self.assertCountEqual(self.GA.mutation(
            child2, self.GA.MUTATION_RATE), child2)

    def test_mutation_return_replace_txpower_randomly(self) -> None:
        selected_population = self.GA.selection(
            self.POPULATION, self.FITNESS[0])

        parent1, parent2 = selected_population[0], selected_population[-1]

        child1, child2 = self.GA.crossover(parent1, parent2)

        self.assertTrue(self.GA.mutation([child1, child2], 10.0, method='random') == [
                        child1, child2] or self.GA.mutation([child1, child2], 10.0, method='random') != [child1, child2])

    def test_mutation_random_mode_returns_three_elements_in_a_tuple(self) -> None:
        selected_population = self.GA.selection(
            self.POPULATION, self.FITNESS[0])

        parent1, parent2 = selected_population[0], selected_population[-1]

        child1, child2 = self.GA.crossover(parent1, parent2)

        self.assertEqual(len(self.GA.mutation(
            [child1, child2], 10.0, method='random')[0]), 3)

        self.assertEqual(len(self.GA.mutation(
            [child1, child2], 10.0, method='random')[-1]), 3)

    def test_run_return_a_tuple(self) -> None:
        self.assertIsInstance(self.GA.run(
            self.POPULATION, self.FITNESS), tuple)

    def test_run_return_population_and_fitness(self) -> None:
        self.assertEqual(len(self.GA.run(self.POPULATION, self.FITNESS)), 2)

    def test_run_return_population_and_fitness_with_given_population_length(self) -> None:
        self.assertEqual(
            len(self.GA.run(self.POPULATION, self.FITNESS)[0]), 10)

        self.assertEqual(
            len(self.GA.run(self.POPULATION, self.FITNESS)[1]), 10)

    def test_run_return_population_does_not_include_given_population_in_same_order(self) -> None:
        self.assertNotEqual(self.GA.run(
            self.POPULATION, self.FITNESS)[0], self.POPULATION)

    def test_run_return_fitness_does_not_include_given_fitness_in_same_order(self) -> None:
        self.assertNotEqual(self.GA.run(
            self.POPULATION, self.FITNESS)[1], self.FITNESS)


if __name__ == '__main__':
    unittest.main()
