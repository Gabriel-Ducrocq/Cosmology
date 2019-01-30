import numpy as np
from sampler import Sampler
from utils import RBF_kernel, compute_discrepency
import pickle
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

NSIDE = 4
sigma_rbf = 100000
N_PROCESS_MAX = 45
N_sample = 10000


def pipeline(tuple_input):
    sampler, reference_data = tuple_input
    sampled = sampler.sample_model()
    sim_data = sampled["sky_map"]
    discrepency = compute_discrepency(reference_data, sim_data)
    probas = RBF_kernel(discrepency, sigma_rbf)
    accepted = np.random.binomial(1, probas)
    return {"sky_map": sim_data, "cosmo_params": sampled["cosmo_params"], "betas": sampled["betas"], "proba": probas,
                "accepted": accepted, "discrepency":discrepency}


def main(NSIDE):
    reference_data = np.load("B3DCMB/reference_data.npy")
    sampler = Sampler(NSIDE)
    time_start = time.time()
    pool = mp.Pool()
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

if __name__=='__main__':
    main(NSIDE)