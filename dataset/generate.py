import numpy as np
from configs.parse_args import parse_args

if __name__ == "__main__":
    config = parse_args()

    np.random.seed(0)
    low = 0.0
    high = 5.0
    uniform_data = np.random.uniform(low, high, size=(config.batch_size, config.input_dim))

    np.save("data.npy", uniform_data)
