from ..basis.Lipkin import Lipkin
from ..HF.HF import HF
from ..CC.CCD import CCD
from ..CC.CCSD import CCSD

import numpy as np

def main():
    basis = Lipkin(2, eps=1.0, V=1.0)

    print(np.count_nonzero(basis.v))
    print(np.array(np.nonzero(basis.v)).T)
    print(basis.v[np.nonzero(basis.v)])
    print(basis.is_AS_)


    print(basis.evaluate_energy())

    hf = HF(basis)

    hf.run()

    print(
        hf.evaluate_energy()
    )

if __name__ == '__main__':
    main()