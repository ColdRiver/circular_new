import numpy as np
from dataclasses import dataclass

## Settings for the stages (Leader acts first, then Followers sequence)
LEADER = 0
BUYER = 1
TRANSFORM = 2
stages = {LEADER, BUYER, TRANSFORM}

config = {
    # System parameters
    'num_agents': 3,
    'num_commodities': 12,
    
    # Physical/Chemical Parameters
    'alpha': 0.5,
    'beta': 1.5,
    'delta': 0.5,
    'LAMBDA': 0.5,
    'UC': 0.5,
    'TX_P': 0.5,
    'INIT_INV': 100,
    'RWD_SCALE': 1e-9,
    
    # Bilevel Reinforcement Learning (BRL) Hyperparameters (Gaur et al. 2025)
    'lambda_penalty': 0.5,         # Penalty coefficient (lambda) for value gap
    'lr_leader': 3e-4,             # Slower learning rate for the Upper-Level Leader (phi)
    'lr_follower': 3e-4,           # Faster learning rate for the Lower-Level Followers (theta)
    'leader_update_frequency': 5,  # Slower timescale ratio (updates leader once every N epochs)
    
    # Training parameters
    'gamma': 0.99,
    'num_steps': 1000,             # Number of steps per epoch
    'episode_length': 1000,
    'num_epochs': 100,
    'history_length': 5,
    'save_freq': 1,
    'seed': 2024,
    'price_factor': 1.,
}

def init_historical_data():
    historic_data = {}
    historic_data['spot_price'] = np.array([
        [config['price_factor']*0.5], [0.8], [1.], [3.], [20.], [4.], 
        [8.], [100.], [0.2], [1.2], [0.15], [1.173]
    ])
    return historic_data
