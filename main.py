import numpy as np
from sampler import Sampler
from utils import RBF_kernel, compute_discrepency
import pickle
import multiprocessing as mp
import time
import dill

NSIDE = 4
sigma_rbf = 100000
N_sample = 60


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
    all_results = []
    reference_data = np.load("B3DCMB/reference_data.npy")
    sampler = Sampler(NSIDE)
    time_start = time.clock()
    '''
    processes = [mp.Process(target=pipeline, args=(sampler, reference_data, output, )) for _ in range(N_sample)]
    for i in range(int(np.floor(N_sample/N_PROCESS_MAX))):
        for p in processes:
            p.start()

        results = [output.get() for p in processes]
        for p in processes:
            p.join()

        all_results += results
        print("ENDENDENDENDENDEND")

    '''
    pool = mp.Pool()
    all_results = pool.map(pipeline, ((sampler, reference_data, ) for _ in range(N_sample)))
    time_elapsed = time.clock() - time_start

    with open("B3DCMB/results", "wb") as f:
        pickle.dump(all_results, f)


    print(time_elapsed)
    print(len(all_results))

if __name__=='__main__':
    main(NSIDE)