import numpy as np 


if __name__ == '__main__':
    np.random.seed(1)

    print(np.random.rand(4))
    np.random.seed(1)

    print(np.random.rand(4))
    np.random.seed()

    print(np.random.rand(4))
