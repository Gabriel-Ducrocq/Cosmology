import numpy as np
from sampler import Sampler
from utils import RBF_kernel, compute_discrepency
import pickle
import multiprocessing as mp
import time

NSIDE = 4
sigma_rbf = 100000
N_sample = 100



def pipeline(sampler, reference_data, output):
    sampled = sampler.sample_model()
    sim_data = sampled["sky_map"]
    discrepency = compute_discrepency(reference_data, sim_data)
    probas = RBF_kernel(discrepency, sigma_rbf)
    accepted = np.random.binomial(1, probas)

    output.put({"sky_map": sim_data, "cosmo_params": sampled["cosmo_params"], "betas": sampled["betas"], "proba": probas,
                "accepted": accepted, "discrepency":discrepency})

def main(NSIDE):
    reference_data = np.load("B3DCMB/reference_data.npy")
    output = mp.Queue()
    sampler = Sampler(NSIDE)
    time_start = time.clock()
    processes = [mp.Process(target=pipeline, args=(sampler, reference_data, output, )) for _ in range(N_sample)]
    for p in processes:
        p.start()

    results = [output.get() for p in processes]
    for p in processes:
        p.join()

    time_elapsed = time.clock() - time_start

    with open("B3DCMB/results", "wb") as f:
        pickle.dump(results, f)

    print(time_elapsed)

if __name__=='__main__':
    main(NSIDE)