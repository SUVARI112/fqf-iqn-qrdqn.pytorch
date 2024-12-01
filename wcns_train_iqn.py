import os
import yaml
import argparse
from datetime import datetime

import numpy as np
from fqf_iqn_qrdqn.environment.environment import Environment, Environment_eval
from fqf_iqn_qrdqn.environment.plant import Plant

from fqf_iqn_qrdqn.env import make_pytorch_env
from fqf_iqn_qrdqn.agent import IQNAgent


def run(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    
    # define Environment related parameters (controllability, no of used frequencies and plants)
    controllability = 2
    no_of_channels = 3

    plants = [
        Plant(2,
              controllability,
              np.array([[1.1, 0.2], [0.2, 0.8]]),
              np.array([[1], [1]]),
              np.identity(2),
              0.1 * np.identity(2),
              0.1 * np.identity(2),
              np.array([[-2.900, 1.000]])),
        Plant(2,
              controllability,
              np.array([[1.2, 0.2], [0.2, 0.9]]),
              np.array([[1], [1]]),
              np.identity(2),
              0.1 * np.identity(2),
              0.1 * np.identity(2),
              np.array([[-3.533, 1.433]])),
        Plant(2,
              controllability,
              np.array([[1.2, 0.2], [0.2, 0.9]]),
              np.array([[1], [1]]),
              np.identity(2),
              0.1 * np.identity(2),
              0.1 * np.identity(2),
              np.array([[-3.533, 1.433]])),
        Plant(2,
              controllability,
              np.array([[1.3, 0.2], [0.2, 1.0]]),
              np.array([[1], [1]]),
              np.identity(2),
              0.1 * np.identity(2),
              0.1 * np.identity(2),
              np.array([[-4.233, 1.933]])),
        # Plant(2,
        #       controllability,
        #       np.array([[1.3, 0.2], [0.2, 1.0]]),
        #       np.array([[1], [1]]),
        #       np.identity(2),
        #       0.1 * np.identity(2),
        #       0.1 * np.identity(2),
        #       np.array([[-4.233, 1.933]]))
    ]

    # Generate random success rates for uplink and downlink channels, fix the random seed to check the results later.
    np.random.seed(112)
    uplink_coefficients = np.random.uniform(0.65, 1, (no_of_channels, len(plants)))
    downlink_coefficients = np.random.uniform(0.65, 1, (no_of_channels, len(plants)))
    # env = Environment(plants, no_of_channels, uplink_coefficients, downlink_coefficients, controllability)

    # Create environments.
    env = Environment(plants, no_of_channels, uplink_coefficients, downlink_coefficients, controllability)
    test_env = Environment_eval(plants, no_of_channels, uplink_coefficients, downlink_coefficients, controllability)

    # Specify the directory to log.
    name = args.config.split('/')[-1].rstrip('.yaml')
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        'logs', args.env_id, f'{name}-seed{args.seed}-{time}')

    # Create the agent and run.
    agent = IQNAgent(
        env=env, test_env=test_env, log_dir=log_dir, seed=args.seed,
        cuda=args.cuda, **config)
    agent.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', type=str, default=os.path.join('config', 'iqn.yaml'))
    parser.add_argument('--env_id', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=142)
    args = parser.parse_args()
    run(args)
