import numpy as np
from sampler import Sampler
from utils import RBF_kernel, compute_discrepency_L2, compute_discrepency_Inf, compute_acceptance_rates, \
    histogram_posterior, graph_dist_vs_theta, graph_dist_vs_dist_theta
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

NSIDE = 4
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
    '''
    reference_data = np.load("B3DCMB/reference_data_extrem.npy")
    sampler = Sampler(NSIDE)

    time_start = time.time()
    pool = mp.Pool(N_PROCESS_MAX)
    all_results = pool.map(pipeline, ((sampler, reference_data, ) for _ in range(N_sample)))
    time_elapsed = time.time() - time_start
    print(time_elapsed)

    with open("B3DCMB/results_extrem", "wb") as f:
        pickle.dump(all_results, f)

    discr_L2 = []
    discr_Inf = []
    for res in all_results:
        discr_L2.append(res["discrepency_L2"])
        discr_Inf.append(res["discrepency_Inf"])

    plt.hist(discr_L2)
    plt.title("Discrepencies for L2 distance")
    plt.savefig("B3DCMB/hist_discr_L2_extrem.png")
    plt.close()

    plt.hist(discr_Inf)
    plt.title("Discrepencies for norm sup distance")
    plt.savefig("B3DCMB/hist_discr_normsup_extrem.png")
    plt.close()

    '''
    '''
    discrepencies = []
    for dico in all_results:
        discrepencies.append(dico["discrepency"])


    plt.hist(discrepencies, density = True)
    plt.savefig('B3DCMB/discrepencies_histogram.png')

    print(np.mean(discrepencies))
    print(np.median(discrepencies))
    '''

    with open("B3DCMB/results", "rb") as f:
        results = pickle.load(f)

    reference_cosmo = np.load("B3DCMB/reference_cosmo.npy")
    reference_betas = np.load("B3DCMB/reference_beta.npy")


    discrepencies_l2 = []
    cosmo_sample = []
    betas = []
    for res in results:
        discrepencies_l2.append(res["discrepency"])
        cosmo_sample.append(res["cosmo_params"])
        betas.append(res["betas"])

    graph_dist_vs_dist_theta(discrepencies_l2, cosmo_sample, reference_cosmo)


    '''
    plt.plot(epsilons, means)
    plt.savefig("B3DCMB/acceptance_ratio_vs_epsilon.png")
    '''


if __name__=='__main__':
    main(NSIDE)

