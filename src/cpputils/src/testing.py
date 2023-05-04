import numpy as np
from cpputils import makeAS

def slow(x_old, x_new, L):
    for i in range(L):
        for j in range(L):
            for k in range(L):
                for l in range(L):
                    elm = x_old[i,j,k,l] - x_old[i,j,l,k]
                    x_new[i,j,k,l] = elm
    # for i in range(L):
    #     for j in range(i, L):
    #         for k in range(L):
    #             for l in range(k, L):
    #                 x_new[i, j, k, l] = x_old[i, j, k, l] - x_old[i, j, l, k]

def main():
    L = 10
    x = np.random.uniform(low=0.5, high=1.5, size=(L,L,L,L))
    x_saves = x.copy()

    x_new_slow = np.zeros_like(x)

    slow(x, x_new_slow, L)

    x_new_fast = np.zeros_like(x)
    makeAS(x, x_new_fast)
    np.testing.assert_allclose(x, x_saves, atol=1e-10)

    np.testing.assert_allclose(x_new_fast, x_new_slow, atol=1e-10)
if __name__ == '__main__':
    main()