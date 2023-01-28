import os
import sys
import random
import unittest
import numpy as np
from datetime import datetime
from scipy.spatial.distance import cdist

'''
Add the src/model directory to the path. This is needed because the test is
run from the root directory and the model.py file is in the src/model directory.

Reference: https://www.geeksforgeeks.org/python-import-from-parent-directory/
'''
dir = os.path.join(os.getcwd(), 'src', 'model')
sys.path.append(dir)

from model import Model  # noqa


class TestModel(unittest.TestCase):
    def setUp(self) -> None:
        self.DEVICES: dict = {"Unallocated": 0, "SEED": 1,
                              "REED": 2, "Router": 3, "Leader": 4, "Border Router": 5, }
        self.TOTAL_DEVICE: int = 8

        distance = np.random.random((self.TOTAL_DEVICE, 2))

        self.DISTANCES: np.array = cdist(distance, distance)
        self.FC: float = 2.4e9
        self.D0: float = 0.25
        self.SIGMA: int = 3
        self.EXP: int = 4
        self.GT: int = 1
        self.GR: int = 1
        self.RSSI_THRESHOLD: int = -100
        self.PENALTY: int = 1000

        self.model = Model(devices=self.DEVICES, total_device=self.TOTAL_DEVICE, distances=self.DISTANCES, fc=self.FC, d0=self.D0,
                           sigma=self.SIGMA, exp=self.EXP, gt=self.GT, gr=self.GR, rssi_threshold=self.RSSI_THRESHOLD, penalty=self.PENALTY)

        self.nodes = [random.choice(list(self.model.DEVICES.values()))
                      for _ in range(self.model.TOTAL_DEVICE)]
        txpower = [random.choice([_ for _ in range(-20, self.model.TOTAL_DEVICE+1, 4)])
                   for _ in range(self.model.TOTAL_DEVICE)]
        x = [(node, position, transmission_power) for position,
             (node, transmission_power) in enumerate(zip(self.nodes, txpower))]
        self.important_nodes = sorted(x, key=lambda x: x[0], reverse=True)

        return super().setUp()

    def test_path_loss_when_distance_is_0(self):
        np.testing.assert_equal(self.model.lognorm(
            self.FC, self.DISTANCES[0][0], self.D0, self.EXP, self.SIGMA), np.nan)

    def test_path_loss_return_float(self):
        self.assertIsInstance(self.model.lognorm(
            self.FC, np.float64(1), self.D0, self.EXP, self.SIGMA), float)

    def test_path_loss_return_np_ndarray(self):
        self.assertIsInstance(self.model.lognorm(
            self.FC, self.DISTANCES, self.D0, self.EXP, self.SIGMA), np.ndarray)

    def test_path_loss_when_noise_is_true(self):
        self.assertIsInstance(self.model.lognorm(
            self.FC, np.float64(1), self.D0, self.EXP, self.SIGMA, noise=True), np.ndarray)

        self.assertIsInstance(self.model.lognorm(
            self.FC, np.float64(1), self.D0, self.EXP, self.SIGMA, noise=True)[0], float)

    def test_path_loss_when_noise_is_false(self):
        self.assertIsInstance(self.model.lognorm(
            self.FC, np.float64(1), self.D0, self.EXP, self.SIGMA, noise=False), float)

    def test_path_loss_when_noise_is_not_specified(self):
        self.assertIsInstance(self.model.lognorm(
            self.FC, np.float64(1), self.D0, self.EXP, self.SIGMA), float)

    def test_path_loss_when_noise_is_not_boolean(self):
        self.assertIsInstance(self.model.lognorm(
            self.FC, np.float64(1), self.D0, self.EXP, self.SIGMA, noise="True"), np.ndarray)

        self.assertIsInstance(self.model.lognorm(
            self.FC, np.float64(1), self.D0, self.EXP, self.SIGMA, noise="False"), np.ndarray)

        self.assertIsInstance(self.model.lognorm(
            self.FC, np.float64(1), self.D0, self.EXP, self.SIGMA, noise="False")[0], float)

    def test_path_loss_for_non_zero_distances(self):
        self.assertEqual(np.all(self.DISTANCES[self.DISTANCES != 0]), True)

    def test_path_loss_for_zero_unique_distances(self):
        self.assertEqual(
            len(np.unique(self.DISTANCES[self.DISTANCES != 0])), 28)

    def test_replace_path_loss_nan_with_mean(self):
        path_loss = self.model.lognorm(
            self.FC, self.DISTANCES, self.D0, self.EXP, self.SIGMA)

        path_loss[np.isnan(path_loss)] = np.nanmean(path_loss)

        self.assertEqual(np.all(path_loss[path_loss != np.nan]), True)

    def test_replace_path_loss_nan_to_zero(self):
        '''
        This test is to check if the path loss is replaced with zero when the distance is zero.

        Reference: https://note.nkmk.me/en/python-numpy-nan-replace/
        '''
        path_loss = self.model.lognorm(
            self.FC, self.DISTANCES, self.D0, self.EXP, self.SIGMA)

        self.assertEqual(np.all(np.nan_to_num(path_loss) != np.nan), True)

    def test_calculate_rssi_return_tuple(self):
        path_loss = self.model.calculate_path_loss(
            current_position=5, next_position=7)

        self.assertIsInstance(self.model.calculate_rssi(
            current_transmission_power=2, next_transmission_power=3, path_loss=path_loss), tuple)

    def test_calculate_rssi_return_no_nan(self):
        path_loss = self.model.calculate_path_loss(
            current_position=0, next_position=0)

        rssi_downlink, rssi_uplink = self.model.calculate_rssi(
            current_transmission_power=0, next_transmission_power=0, path_loss=path_loss)

        sensitivity_penalty = self.model.rssi_sensitivity_penalty(
            rssi_downlink, rssi_uplink)

        self.assertEqual(np.isnan(rssi_downlink), False)
        self.assertEqual(np.isnan(rssi_uplink), False)
        self.assertEqual(np.isnan(sensitivity_penalty), False)

    def test_penalized_transmission_power_from_rssi_calculation(self):
        model = Model(distances=np.array([[123456789]]))

        path_loss = model.calculate_path_loss(
            current_position=0, next_position=0)

        rssi_downlink, rssi_uplink = model.calculate_rssi(
            current_transmission_power=0, next_transmission_power=0, path_loss=path_loss)

        sensitivity_penalty = model.rssi_sensitivity_penalty(
            rssi_downlink, rssi_uplink)

        self.assertLess(rssi_downlink, -100)
        self.assertLess(rssi_uplink, -100)
        self.assertEqual(sensitivity_penalty, self.PENALTY * 2)

    def test_grouper_for_specific_chunks(self):
        self.assertEqual(
            len(list(self.model.grouper(iter([_ for _ in range(10)]), 3))), 4)

    def test_grouper_for_list_values(self):
        self.assertIsInstance(list(self.model.grouper(
            iter([_ for _ in range(10)]), 3)), list)

    def test_grouper_for_tuple_values(self):
        self.assertIsInstance(tuple(self.model.grouper(
            iter([_ for _ in range(10)]), 3)), tuple)

    def test_grouper_for_list_values_equal(self):
        self.assertListEqual(list(self.model.grouper(iter([_ for _ in range(10)]), 3)), [
                             [0, 1, 2], [3, 4, 5], [6, 7, 8], [9]])

    def test_combine_network_topology_return_a_list(self):
        self.assertIsInstance(self.model.combine_network_topology(
            self.important_nodes, unallocated=True), list)

    def test_combine_network_topology_return_a_list_of_tuples(self):
        self.assertIsInstance(self.model.combine_network_topology(
            self.important_nodes, unallocated=True)[0], tuple)

    def test_combine_network_topology_return_a_list_of_tuples_with_length_7(self):
        self.assertEqual(len(self.model.combine_network_topology(
            self.important_nodes, unallocated=True)[0]), 7)

    def test_combine_network_topology_unallocated_mode_enabled_to_filter_out_unallocated_nodes(self):
        nodes = list(self.model.DEVICES.values()) + [0, 5, 3]
        txpower = [random.choice([_ for _ in range(-20, self.model.TOTAL_DEVICE+1, 4)])
                   for _ in range(self.model.TOTAL_DEVICE)]
        x = [(node, position, transmission_power) for position,
             (node, transmission_power) in enumerate(zip(nodes, txpower))]
        important_nodes = sorted(x, key=lambda x: x[0], reverse=True)

        specification = self.model.combine_network_topology(
            important_nodes, unallocated=False)

        for spec in specification:
            current_device = spec[0]
            next_device = spec[1]

            self.assertNotEqual(current_device, 0)
            self.assertNotEqual(next_device, 0)

    def test_triangle_combine_network_topology_return_a_list(self):
        self.assertIsInstance(
            self.model.triangle_combine_network_topology(self.important_nodes), list)

    def test_triangle_combine_network_topology_return_a_list_of_lists(self):
        self.assertIsInstance(self.model.triangle_combine_network_topology(
            self.important_nodes)[0], list)

    def test_triangle_combine_network_topology_return_a_list_of_tuples(self):
        self.assertIsInstance(self.model.triangle_combine_network_topology(
            self.important_nodes)[0][0], tuple)

    def test_triangle_combine_network_topology_return_a_list_of_tuples_with_length_7(self):
        self.assertEqual(len(self.model.triangle_combine_network_topology(
            self.important_nodes)[0][0]), 7)

    def test_mathematical_constraints_penalty_returns_a_tuple(self):
        self.assertIsInstance(self.model.mathematical_constraints_penalty(
            self.nodes), tuple)

    def test_mathematical_constraints_penalty_returns_a_tuple_with_length_2(self):
        self.assertEqual(len(self.model.mathematical_constraints_penalty(
            self.nodes)), 2)

    def test_mathematical_constraints_penalty_returns_a_tuple_with_penalty(self):
        nodes = [5, 5, 4, 3, 3, 2, 2, 1]

        self.assertNotEqual(
            self.model.mathematical_constraints_penalty(nodes)[1], 0)

    def test_mathematical_constraints_penalty_returns_a_tuple_with_no_penalty(self):
        nodes = [5, 5, 4, 3, 3, 2, 2, 2]

        self.assertEqual(
            self.model.mathematical_constraints_penalty(nodes)[1], 0)

    def test_mathematical_constraints_penalty_returns_a_tuple_with_valid_device_types_list(self):
        valid_nodes = [5, 5, 4, 3, 3, 2, 2, 2]

        self.assertListEqual(self.model.mathematical_constraints_penalty(
            valid_nodes)[0], valid_nodes)

    def test_plot_distance_stored_in_file(self):
        path: str = os.path.join(os.getcwd(), 'dist', "distance.png")

        self.model.plot_distance(self.DISTANCES, path)

        self.assertTrue(path)
        self.assertTrue(os.path.exists(path))

        os.remove(path)

        self.assertFalse(os.path.exists(path))

    def test_plot_path_loss_model_stored_in_file(self):
        path: str = os.path.join(os.getcwd(
        ), 'dist', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-path-loss')}.png")

        self.model.plot_path_loss_model(path)

        self.assertTrue(path)
        self.assertTrue(os.path.exists(path))

        os.remove(path)

        self.assertFalse(os.path.exists(path))

    def test_plot_path_loss_model_stored_in_file_with_noise(self):
        path: str = os.path.join(os.getcwd(
        ), 'dist', f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-path-loss')}.png")

        self.model.plot_path_loss_model(path, noise=True)

        self.assertTrue(path)
        self.assertTrue(os.path.exists(path))

        os.remove(path)

        self.assertFalse(os.path.exists(path))


if __name__ == '__main__':
    unittest.main()
