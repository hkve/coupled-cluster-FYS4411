import numpy as np

def amplitudes_ccd(t, v, f_pp_o, f_hh_o, vir, occ):
    res = np.zeros_like(t)

    res += v[vir, vir, occ, occ] # v_abij

    tp = np.einsum("bc,acij->abij", f_pp_o, t, optimize=True)
    res += (tp - tp.transpose(1,0,2,3))

    tp = np.einsum("kj,abik->abij", f_hh_o, t, optimize=True)
    res -= (tp - tp.transpose(0,1,3,2))

    # Two first sums, over cd and kl
    res += 0.5*np.einsum("abcd,cdij->abij", v[vir, vir, vir, vir], t, optimize=True)
    res += 0.5*np.einsum("klij,abkl->abij", v[occ, occ, occ, occ], t, optimize=True)

    # First permutation term, P(ij|ab), over kc
    tp = np.einsum("kbcj,acik->abij", v[occ, vir, vir, occ], t, optimize=True)
    res += (tp - tp.transpose(1,0,2,3) - tp.transpose(0,1,3,2) + tp.transpose(1,0,3,2))

    # First double t sum, over klcd
    res += 0.25*np.einsum("klcd,cdij,abkl->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    
    # First P(ij) permutation, double t sum over klcd
    tp = np.einsum("klcd,acik,bdjl->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res += (tp - tp.transpose(0,1,3,2))

    # second P(ij) permutation, double t sum over klcd
    tp = np.einsum("klcd,dcik,ablj->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res -= 0.5*(tp - tp.transpose(0,1,3,2))

    # Only P(ab) term, double t sum over klcd
    tp = np.einsum("klcd,aclk,dbij->abij", v[occ, occ, vir, vir], t, t, optimize=True)
    res -= 0.5*(tp - tp.transpose(1,0,2,3))
    
    return res