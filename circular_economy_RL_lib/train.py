import sys
import os
import multiprocessing
from trainer import BilevelTrainer
from utils import create_logger
import os
import sys

# Prevent OpenMP and MKL thread-pool deadlocks between PyTorch and TensorFlow
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Ensure PYTHONPATH handles local modules correctly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger_params = {
    'log_file': {
        'desc': 'bilevel_train',
        'filename': 'run_log'
    }
}

def main():
    create_logger(**logger_params)
    print(f"System CPUs available: {multiprocessing.cpu_count()}")
    print("Initializing Gaur et al. (2025) Bilevel Reinforcement Learning...")
    
    # Instantiate the hierarchical trainer
    trainer = BilevelTrainer()
    
    # Launch training
    trainer.learn()

if __name__ == "__main__":
    main()
