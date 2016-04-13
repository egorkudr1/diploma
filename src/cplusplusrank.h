#ifndef __RANK_H__
#define __RANK_H__


void 
fast_climf_fit(double* U, double* V, int N_user, int N_item, int K, int * edge_u, int* edge_i, int len_edge, 
                        double reg, double lrate, int maxiter);

void 
fast_bpr_mf_fit(double* U, double* V, int N_user, int N_item, int K, int * edge_u, int* edge_i, int len_edge, 
                        double regU, double regIpos, double regIneg, double lrate, int maxiter);

void
fast_tfmap_fit(double* U, double* V, int N_user, int N_item, int K, int * edge_u, int* edge_i, int len_edge,
                    double reg, double lrate, int n_sample, int maxiter);


#endif