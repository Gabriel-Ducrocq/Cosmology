import numpy as np
from sampler import Sampler
from utils import RBF_kernel, compute_discrepency_L2, compute_discrepency_Inf
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
    discrepency = compute_discrepency_L2(reference_data, sim_data)
    probas = RBF_kernel(discrepency, sigma_rbf)
    accepted = np.random.binomial(1, probas)
    return {"sky_map": sim_data, "cosmo_params": sampled["cosmo_params"], "betas": sampled["betas"], "proba": probas,
            "accepted": accepted, "discrepency":discrepency}


def main(NSIDE):
    '''
    reference_data = np.load("B3DCMB/reference_data.npy")
    sampler = Sampler(NSIDE)
    time_start = time.time()
    pool = mp.Pool(N_PROCESS_MAX)
    all_results = pool.map(pipeline, ((sampler, reference_data, ) for _ in range(N_sample)))
    time_elapsed = time.time() - time_start
    print(time_elapsed)

    with open("B3DCMB/results", "wb") as f:
        pickle.dump(all_results, f)

    discrepencies = []
    for dico in all_results:
        discrepencies.append(dico["discrepency"])


    plt.hist(discrepencies, density = True)
    plt.savefig('B3DCMB/discrepencies_histogram.png')

    print(np.mean(discrepencies))
    print(np.median(discrepencies))
    '''

    '''
    with open("B3DCMB/results", "rb") as f:
        results = pickle.load(f)

    epsilon = 6e6

    discrepencies = []
    cosmo_sample = []
    beta_sample = []
    for res in results:
        discrepencies.append(res["discrepency"])
        cosmo_sample.append(res["cosmo_params"])
        beta_sample.append(res["betas"])

    probas = RBF_kernel(np.array(discrepencies), epsilon)
    accepted = np.random.binomial(1, probas)
    print(np.mean(accepted))
    accepted_cosmo = [l[1] for l in list(zip(accepted, cosmo_sample)) if l[0] == 1]
    print("Only one kept")
    print(len(accepted_cosmo))
    reference_cosmo = np.load("B3DCMB/reference_cosmo.npy")

    for i, name in enumerate(COSMO_PARAMS_NAMES):
        print(i)
        e = []
        for set_cosmos in cosmo_sample:
            e.append(set_cosmos[i])

        print(np.mean(e))
        print(np.median(e))
        prior = np.random.normal(COSMO_PARAMS_MEANS[i], COSMO_PARAMS_SIGMA[i], 10000)
        plt.hist(prior, density=True, alpha=0.5, label="Prior")
        plt.hist(e, density = True, alpha = 0.5, label = "ABC posterior", weights = probas)
        plt.legend(loc='upper right')
        plt.title('Histogram parameter: '+name)
        plt.axvline(reference_cosmo[i], color='k', linestyle='dashed', linewidth=1)
        plt.savefig("B3DCMB/histogram_" + name + ".png")
        plt.close()

    '''


    '''
    plt.plot(epsilons, means)
    plt.savefig("B3DCMB/acceptance_ratio_vs_epsilon.png")
    '''
    reference_data = np.load("B3DCMB/reference_data.npy")
    with open("B3DCMB/results", "rb") as f:
        results = pickle.load(f)

    sky_maps = []
    cosmo_params = []
    betas = []
    discrepencies_inf = []
    for res in results:
        sky_maps.append(res["sky_map"])
        cosmo_params.append(res["cosmo_params"])
        betas.append(res["betas"])
        discrepencies_inf.append(res["discrepency_inf"])


    epsilon = np.linspace(2000, 3000, 1000)
    acceptance_ratio = []
    for eps in epsilon:
        acceptance_ratio.append(np.mean(np.random.binomial(1, RBF_kernel(np.array(discrepencies_inf),eps))))

    plt.plot(epsilon, acceptance_ratio)
    plt.savefig("B3DCMB/acceptance_ratio_vs_epsilon_inf.png")


if __name__=='__main__':
    main(NSIDE)