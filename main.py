import numpy as np
from sampler import Sampler
from utils import RBF_kernel, compute_discrepency_L2, compute_discrepency_Inf, compute_acceptance_rates, \
    histogram_posterior, graph_dist_vs_theta, graph_dist_vs_dist_theta
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import time
from scipy import stats

NSIDE = 1
sigma_rbf = 100000
N_PROCESS_MAX = 45
N_sample = 5000

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
    reference_data = np.load("data/reference_values/reference_data_simplified.npy")
    sampler = Sampler(NSIDE)

    time_start = time.time()
    pool = mp.Pool(N_PROCESS_MAX)
    all_results = pool.map(pipeline, ((sampler, reference_data, ) for _ in range(N_sample)))
    time_elapsed = time.time() - time_start
    print(time_elapsed)
    with open("data/simulations/results_inf", "wb") as f:
        pickle.dump(all_results, f)

    '''
    with open("data/simulations/results_inf", "rb") as f:
        all_results_inf = pickle.load(f)

    with open("data/simulations/results_sup", "rb") as f:
        all_results_sup = pickle.load(f)




    sky_maps_inf = []
    sky_maps_sup = []
    for res in all_results_sup:
        sky_maps_sup.append(res["sky_map"].flatten().tolist())

    for res in all_results_inf:
        sky_maps_inf.append(res["sky_map"].flatten().tolist())

    by_pixels_sup = list(zip(*sky_maps_sup))
    by_pixels_inf = list(zip(*sky_maps_inf))

    print(len(by_pixels_sup))
    mat_corr_sup = np.zeros((12*NSIDE, 12*NSIDE))
    mat_corr_inf = np.zeros((12 *NSIDE, 12*NSIDE))
    for i, l1 in enumerate(by_pixels_sup):
        for j, l2 in enumerate(by_pixels_sup):
            print(i)
            print(j)
            print(len(l1))
            print(len(l2))
            corr = stats.pearsonr(l1, l2)
            mat_corr_sup[i,j] = corr[0]

    for i, l1 in enumerate(by_pixels_inf):
        for j, l2 in enumerate(by_pixels_inf):
            corr = stats.pearsonr(l1, l2)
            mat_corr_inf[i,j] = corr[0]




    '''
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
    '''
    with open("data/simulations/results_simplified", "rb") as f:
        all_results = pickle.load(f)

    discr_L2 = []
    discr_Inf = []
    cosmo_sample = []
    for res in all_results:
        discr_L2.append(res["discrepency_L2"])
        discr_Inf.append(res["discrepency_Inf"])
        cosmo_sample.append(res["cosmo_params"])

    epsilon_inf = 1.1*(1e-8)
    epsilon_l2 = 1*(1e-14)

    reference_cosmo = np.load("data/reference_values/reference_cosmo_simplified.npy")

    histogram_posterior(epsilon_l2, discr_L2, cosmo_sample, reference_cosmo, "l2_simplified")
    histogram_posterior(epsilon_inf, discr_Inf, cosmo_sample, reference_cosmo, "inf_simplified")

    graph_dist_vs_theta(discr_L2, cosmo_sample, reference_cosmo)
    graph_dist_vs_dist_theta(discr_L2, cosmo_sample, reference_cosmo)
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



if __name__=='__main__':
    main(NSIDE)

