#include "cplusplusrank.h"

#include <vector>
#include <set>
#include <random>
#include <math.h>
#include <algorithm>

using namespace std;
    
inline double sigma(double x) {
    return 1 / (1 + exp(-x));
}

inline double dsigma(double x) {
    return 1 / (2 + exp(-x) + exp(x));
}

void 
fast_climf_fit(double* U, double* V, int N_user, int N_item, int K, int * edge_u, int* edge_i, int len_edge,
                        double reg, double lrate, int maxiter) {
    
    vector<vector<int> > data = vector<vector<int> >(N_user, vector<int>(0));
    for (int i = 0; i < len_edge; ++i) {
        data[edge_u[i]].push_back(edge_i[i]);
    } 

    vector<double> f = vector<double>(N_item, 0);

    for (int iter = 0; iter < maxiter; ++iter) {
        for (int u = 0; u < N_user; ++u) {
            // compute derivative of U
            vector<double> delta_U = vector<double>(K, 0);
            int u_items_size = int(data[u].size());

            for (int i = 0; i < u_items_size; ++i) {
                f[i] = 0;
                for (int k = 0; k < K; ++k) {
                    f[i] += U[K * u + k] * V[K * data[u][i] + k];
                }
            }
            
            for (int j = 0; j < u_items_size; ++j) {
                double coef = sigma(-f[j]);
                for (int k = 0; k < K; ++k) {
                    delta_U[k] +=  coef * V[K * data[u][j] + k];
                }

                for (int i = 0; i < u_items_size; ++i) {
                    double coef = dsigma(f[i] - f[j]) / (1 - sigma(f[i] - f[j]));
                    for (int k = 0; k < K; ++ k) {
                        delta_U[k] += coef * (V[K * data[u][j] + k] - V[K * data[u][i] + k]);
                    }
                }
            }

            for (int k = 0; k < K; ++k) {
                U[K * u + k] += lrate * (delta_U[k] - reg * U[K * u + k]);
            }

            // compute derivative of V
            for (int j = 0; j < u_items_size; ++j) {
                for (int i = 0; i < u_items_size; ++i) {
                    f[i] = 0;
                    for (int k = 0; k < K; ++k) {
                        f[i] += U[K * u + k] * V[K * data[u][i] + k];
                    }
                }
                double coef = sigma(-f[j]);
                for (int i = 0; i < u_items_size; ++i) {
                    coef += dsigma(f[j] - f[i]) * (1 / (1 - sigma(f[i] - f[j])) - 
                        1 / (1 - sigma(f[j] - f[i])) );
                }
                for (int k = 0; k < K; ++k) {
                    V[K * data[u][j] + k] += lrate * (coef * U[K * u + k] - reg * V[K * data[u][j] + k]);
                } 
            }
        }
    }
}

void 
fast_bpr_mf_fit(double* U, double* V, int N_user, int N_item, int K, int* edge_u, int* edge_i, int len_edge, 
                    double regU, double regIpos, double regIneg, double lrate, int maxiter) {
    
    vector<vector<int> > data = vector<vector<int> >(N_user, vector<int>(0));
    for (int i = 0; i < len_edge; ++i) {
        data[edge_u[i]].push_back(edge_i[i]);
    } 

    vector<set<int> > data_set = vector<set<int> >(N_user, set<int>());
    for (int i = 0; i < len_edge; ++i) {
        data_set[edge_u[i]].insert(edge_i[i]);
    }

    default_random_engine generator;
    uniform_int_distribution<int> user_distr(0, N_user - 1);
    uniform_int_distribution<int> neg_item_distr(0, N_item - 1);
    
    vector<uniform_int_distribution<int> > pos_item_distr;
    for (int i = 0; i < N_item; ++i) {
        pos_item_distr.push_back(uniform_int_distribution<int>(0, int(data[i].size()) - 1));
    }

    vector<double> delta_U = vector<double>(K);
    vector<double> delta_V_i = vector<double>(K);
    vector<double> delta_V_j = vector<double>(K);

    for (int iter = 0; iter < maxiter; ++iter) {
        for (int l = 0; l < 10 * len_edge; ++l) {
            int u = user_distr(generator);
            int i = pos_item_distr[u](generator);
            int j;
            while(true) {
                j = neg_item_distr(generator);
                if (data_set[u].find(j) == data_set[u].end()) {
                    break;
                }
            }
            double x = 0;
            for (int k = 0; k < K; ++k) {
                x += U[K * u + k] * (V[K * data[u][i] + k] - V[K * j + k]);
            }
            double coef = sigma(-x);
            for (int k = 0; k < K; ++k) {
                delta_U[k] = lrate * (coef * (V[K * data[u][i] + k] - V[K * j + k]) + regU * U[K * u + k]);
                delta_V_i[k] = lrate * (coef * U[K * u + k] + regIpos * V[K * data[u][i] + k]);
                delta_V_j[k] = lrate * (-coef * U[K * u + k] + regIneg * V[K * j + k]);
            }

            for (int k = 0; k < K; ++k) {
                U[K * u + k] += delta_U[k];
                V[K * data[u][i] + k] += delta_V_i[k];
                V[K * j + k] += delta_V_j[k];
            }
        }
    }
}


vector<int>
buffer_constract(double* U, double *V, int u, int N_item, int K, int n_sample, vector<int>& items, set<int> & set_items) {
    int u_items_size = int(items.size());
    vector<double> pos_f = vector<double>(u_items_size, 0);
    for (int i = 0; i < u_items_size; ++i) {
        for (int k = 0; k < K; ++k) {
            pos_f[i] += U[K * u + k] * V[K * items[i]];
        }
    }
    sort(pos_f.begin(), pos_f.end());
    double p_min = pos_f[0];
    vector<pair<double,int> > neg_items;
    for (int i = 0; i < N_item; ++i) {
        double tmp = 0;
        if (set_items.find(i) == set_items.end()) {
            for (int k = 0; k < K; ++k) {
                tmp += U[K * u + k] * V[K * i +  k];
            }
            if (tmp >= p_min) {
                neg_items.push_back(make_pair(tmp, i));
            }
        }
    }
    if (n_sample < int(neg_items.size())) {
        random_shuffle(neg_items.begin(), neg_items.end());
        neg_items = vector<pair<double, int> >(neg_items.begin(), neg_items.begin() + n_sample);
    }
    sort(neg_items.begin(), neg_items.end());
    if (u_items_size < int(neg_items.size())) {
        neg_items = vector<pair<double, int> >(neg_items.begin(), neg_items.begin() + u_items_size);
    }
    vector<int> res = vector<int>(items.begin(), items.end());
    for (int i = 0; i < int(neg_items.size()); ++i) {
        res.push_back(neg_items[i].second);
    }
    return (res);
}


void
fast_tfmap_fit(double* U, double* V, int N_user, int N_item, int K, int * edge_u, int* edge_i, int len_edge,
                    double reg, double lrate, int n_sample, int maxiter) {

    vector<vector<int> > data = vector<vector<int> >(N_user, vector<int>(0));
    for (int i = 0; i < len_edge; ++i) {
        data[edge_u[i]].push_back(edge_i[i]);
    }

    vector<set<int> > set_data = vector<set<int> >(N_user, set<int>());
    for (int i = 0; i < len_edge; ++i) {
        set_data[edge_u[i]].insert(edge_i[i]);
    } 

    vector<double> f = vector<double>(N_item);
    vector<double> df = vector<double>(N_item);
    vector<double> delta_V = vector<double>(K);
    vector<double> delta_U = vector<double>(K);

    for (int iter = 0; iter < maxiter; ++iter) {
        // compute derivative of U
        for (int u = 0; u < N_user; ++u) {
            int u_items_size = int(data[u].size());
            for (int i = 0; i < u_items_size; ++i) {
                f[i] = 0;
                for (int k = 0; k < K; ++k) {
                    f[i] += V[K * data[u][i] + k] * U[K * u + k];
                }
            }
            
            for (int i = 0; i < u_items_size; ++i) {
                double delta_right = 0, delta_left = 0;
                for (int j = 0; j < u_items_size; ++j) {
                    delta_left += sigma(f[j] - f[i]);
                    delta_right += dsigma(f[j] - f[i]);
                }
                double delta = dsigma(f[i]) * delta_left + sigma(f[i]) * delta_right;
                
                for (int k = 0; k < K; ++k) {
                    delta_V[k] = delta * V[K * data[u][i] + k];
                    for (int j = 0; j < u_items_size; ++j) {
                        delta_V[k] += sigma(f[i]) * dsigma(f[j] -f[i]) * V[K * data[u][j] + k];
                    }
                    delta_V[k] /= u_items_size;
                    delta_V[k] -= reg * U[K * u + k];
                }
            }
        }
        // compute derivative of I
        for (int u = 0; u < N_user; ++u) {
            vector<int> buffer = buffer_constract(U, V, u, N_item, K, n_sample, data[u], set_data[u]);
            int u_items_size = int(buffer.size());
            for (int i = 0; i < u_items_size; ++i) {
                f[i] = 0;
                for (int k = 0; k < K; ++k) {
                    f[buffer[i]] = V[K * buffer[i]  + k] * U[K * u + k];
                }
            }
            for (int i = 0; i < u_items_size; ++i) {
                vector<double> delta_V;
                double coef = 0;
                for (int j = 0; j < int(data[u].size()); ++j) {
                    coef += dsigma(f[i]) * sigma(f[data[u][j]] - f[buffer[i]]) + (sigma(f[data[u][j]]) - 
                        sigma(f[buffer[i]])) * dsigma(f[data[u][j]] - f[buffer[i]]);
                }

                for (int k = 0; k < K; ++k) {
                    delta_V[k] = coef * U[K * u + k] / double(data[u].size()) - reg * V[buffer[i]];
                }
                f[buffer[i]] = 0;
                for (int k = 0; k < K; ++k) {
                    V[K * buffer[i] + k] += lrate * delta_V[k];
                    f[buffer[i]] += V[K * buffer[i] + k] * U[K *u + k];
                }
            } 
        }
    }
}