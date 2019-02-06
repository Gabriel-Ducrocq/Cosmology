import numpy as np
from sampler import Sampler
from utils import RBF_kernel, compute_discrepency_L2, compute_discrepency_Inf, compute_acceptance_rates, \
    histogram_posterior, graph_dist_vs_theta, graph_dist_vs_dist_theta
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

NSIDE = 1
sigma_rbf = 100000
N_PROCESS_MAX = 45
N_sample = 10000

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = [0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]
COSMO_PARAMS_SIGMA = [0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]

def pipeline(tuple_input):
    sampler, reference_data = tuple_input
    sampled = sampler.sample_model()
    sim_data = sampled["sky_map"]
    discrepency_L2 = compute_discrepency_L2((reference_data, sim_data))
    discrepency_inf = compute_discrepency_Inf((reference_data, sim_data))
    return {"sky_map": sim_data, "cosmo_params": sampled["cosmo_params"], "betas": sampled["betas"],
            "discrepency_L2":discrepency_L2, "discrepency_Inf": discrepency_inf}


def main(NSIDE):
    reference_data = np.load("data/reference_values/reference_data_simplified.npy")
    sampler = Sampler(NSIDE)

    time_start = time.time()
    pool = mp.Pool(N_PROCESS_MAX)
    all_results = pool.map(pipeline, ((sampler, reference_data, ) for _ in range(N_sample)))
    time_elapsed = time.time() - time_start
    print(time_elapsed)

    with open("data/simulations/results_simplified", "wb") as f:
        pickle.dump(all_results, f)

    discr_L2 = []
    discr_Inf = []
    cosmo_sample = []
    for res in all_results:
        discr_L2.append(res["discrepency_L2"])
        discr_Inf.append(res["discrepency_Inf"])
        cosmo_sample.append(res["cosmo_params"])

    plt.hist(discr_L2)
    plt.title("Discrepencies for L2 distance simplified model")
    plt.savefig("data/graphics/hist_discr_L2_simplified.png")
    plt.close()

    plt.hist(discr_Inf)
    plt.title("Discrepencies for Inf distance simplified model")
    plt.savefig("data/graphics/hist_discr_Inf_simplified.png")
    plt.close()

    '''
    discrepencies = []
    for dico in all_results:
        discrepencies.append(dico["discrepency"])


    plt.hist(discrepencies, density = True)
    plt.savefig('B3DCMB/discrepencies_histogram.png')

    print(np.mean(discrepencies))
    print(np.median(discrepencies))
    '''



if __name__=='__main__':
    main(NSIDE)

