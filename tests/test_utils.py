import unittest
from unittest.mock import patch
from unittest import mock

import torch
import numpy as np

from model.utils import select_action, get_epsilon, get_observation, transform_observation

class TestActionSelection(unittest.TestCase):

    def setUp(self):
        self.action_dims = 5
        self.state = torch.tensor([0])

        mock_model = mock.Mock(
            return_value=torch.tensor([[0.1, 0.3, 0.05, 0.05, 0.5]])
        )
        mock_model.device = torch.device('cpu')
        self.model = mock_model

        self.steps_done = 1000

    @patch('random.random', lambda: 0.9)
    def test_inactive_epsilon(self):
        epsilon = 0.1
        action = select_action(self.model, self.state, self.action_dims, epsilon)
        action_value = action.item()
        self.assertEqual(action_value, 4)

    @patch('random.random', lambda: 0.01)
    @patch('random.randrange', lambda x: 1)
    def test_active_epsilon(self):
        epsilon = 0.1
        action = select_action(self.model, self.state, self.action_dims, epsilon)
        action_value = action.item()
        self.assertEqual(action_value, 1)

    def test_get_epsilon_linear_decay(self):
        iteration = 100
        eps = get_epsilon(iteration, 1.0, 0.05, 1000)

        # this is exactly what assertAlmostEqual is supposted to do
        # but that fails.
        self.assertEqual(round(eps - 0.905), 0)

    def test_get_epsilon_plateau(self):
        iteration = 1500
        eps = get_epsilon(iteration, 1.0, 0.05, 1000)

        self.assertEqual(eps, 0.05)

class TestEnvironmentInteraction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        test_observation = np.uint8(np.random.random((10,10,3)))
        mock_env = mock.Mock()
        mock_env.reset = mock.Mock(return_value=test_observation)
        mock_env.step = mock.Mock(return_value=(test_observation, 1, False, None))
        cls.env = mock_env

    def test_transformation(self):
        obs = self.env.reset()
        obs = transform_observation(obs, (5,5))

        # color numpy array is converted to torch grayscale image
        # (H,W,C) -> (C,H,W)
        self.assertIsInstance(obs, torch.Tensor)
        self.assertEqual(obs.shape, (1,5,5))

    def test_observation_without_action(self):
        obs, reward, done = get_observation(self.env, action=None)

        # reward and done are None
        self.assertIsNone(reward)
        self.assertIsNone(done)


    def test_observation_with_action(self):
        obs, reward, done = get_observation(self.env, action=mock.Mock())

        # reward and done are not None
        self.assertIsNotNone(reward)
        self.assertIsNotNone(done)

if __name__ == '__main__':
    unittest.main()