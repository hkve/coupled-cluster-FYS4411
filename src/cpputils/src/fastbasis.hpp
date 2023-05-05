#ifndef FASTBASIS_HPP
#define FASTBASIS_HPP

inline int map(int i, int j, int k, int l, int L, int L2, int L3) {
    return i*L3 + j*L2 + k*L + l;
}

void make_AS(double* ptr_old, double* ptr_new, int L) {
    double elm;
    int L2 = L*L;
    int L3 = L*L*L;

    auto mapL = [L, L2, L3](int i, int j, int k, int l) {
        return map(i, j, k, l, L, L2, L3);
    };

    for(int i = 0; i < L; i++) {
        for(int j = i; j < L; j++) {
            for(int k = 0; k < L; k++) {
                for(int l = k; l < L; l++) {
                    elm = ptr_old[mapL(i,j,k,l)] - ptr_old[mapL(i,j,l,k)];
                    ptr_new[mapL(i,j,k,l)] = elm;
                    ptr_new[mapL(i,j,l,k)] = -elm;
                    ptr_new[mapL(j,i,k,l)] = -elm;
                    ptr_new[mapL(j,i,l,k)] = elm;
                }
            }
        }
    }
}

#endif