#ifndef FASTBASIS_HPP
#define FASTBASIS_HPP

inline int map(int i, int j, int k, int l, int L, int L2, int L3) {
    return i*L3 + j*L2 + k*L + l;
}

void makeAS(double* ptr_old, double* ptr_new, int L) {
    double elm;
    int L2 = L*L;
    int L3 = L*L*L;
    for(int i = 0; i < L; i++) {
        for(int j = i; j < L; j++) {
            for(int k = 0; k < L; k++) {
                for(int l = k; l < L; l++) {
                    elm = ptr_old[map(i,j,k,l, L, L2, L3)] - ptr_old[map(i,j,l,k, L, L2, L3)];
                    ptr_new[map(i,j,k,l, L, L2, L3)] = elm;
                    ptr_new[map(i,j,l,k, L, L2, L3)] = -elm;
                    ptr_new[map(j,i,k,l, L, L2, L3)] = -elm;
                    ptr_new[map(j,i,l,k, L, L2, L3)] = elm;
                }
            }
        }
    }
}

#endif