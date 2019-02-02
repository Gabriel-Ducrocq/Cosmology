import numpy as np
import scipy
import healpy as hp
import matplotlib.pyplot as plt

def read_template(path, NSIDE, fields = (0, 1, 2, 3, 4, 5) ):
    map_ = hp.read_map(path, field=fields)
    map_ = hp.ud_grade(map_, nside_out=NSIDE)
    return map_

def compute_Sigma_Q_U(map4, map2, map3, map5):
    return map4 - map2, map3 - map5

def create_mean_var(path, NSIDE):
    map_ = read_template(path, NSIDE)
    Q, U = map_[0], map_[1]
    sigma_Q, sigma_U = compute_Sigma_Q_U(map_[4], map_[2], map_[3], map_[5])
    return Q.tolist(), U.tolist(), (sigma_Q**2).tolist(), (sigma_U**2).tolist()

def get_pixels_params(NSIDE):
    Q_sync, U_sync, sigma_Q_sync, sigma_U_sync = create_mean_var(
        'B3DCMB/COM_CompMap_SynchrotronPol-commander_0256_R2.00.fits', NSIDE)
    Q_dust, U_dust, sigma_Q_dust, sigma_U_dust = create_mean_var(
        'B3DCMB/COM_CompMap_DustPol-commander_1024_R2.00.fits', NSIDE)
    params = {"dust":{"mean":{"Q": Q_dust, "U": U_dust},
                    "sigma":{"Q": sigma_Q_dust, "U": sigma_U_dust}},
             "sync":{"mean":{"Q": Q_sync, "U": U_sync},
                    "sigma":{"Q": sigma_Q_sync, "U": sigma_U_sync}}}

    return params

def get_mixing_matrix_params(NSIDE):
    temp_dust, sigma_temp_dust, beta_dust, sigma_beta_dust = read_template(
        'B3DCMB/COM_CompMap_dust-commander_0256_R2.00.fits', NSIDE, fields =(3,5,6,8))
    beta_sync = hp.read_map('B3DCMB/sync_beta.fits', field=(0))
    beta_sync = hp.ud_grade(beta_sync, nside_out=NSIDE)
    sigma_beta_sync = np.load("B3DCMB/sigma_beta_sync.npy")
    params = {"dust":{"temp":{"mean": temp_dust.tolist(), "sigma":(sigma_temp_dust**2).tolist()},
                      "beta":{"mean":beta_dust.tolist(), "sigma": (sigma_beta_dust**2).tolist()}},
            "sync":{"beta":{"mean":beta_sync.tolist(), "sigma": (sigma_beta_sync**2).tolist()}}}

    return params

def aggregate_pixels_params(params):
    Q_sync = params["sync"]["mean"]["Q"]
    U_sync = params["sync"]["mean"]["U"]
    Q_dust = params["dust"]["mean"]["Q"]
    U_dust = params["dust"]["mean"]["U"]
    templates_map = np.hstack([Q_sync, U_sync, Q_dust, U_dust])
    sigma_Q_sync = params["sync"]["sigma"]["Q"]
    sigma_U_sync = params["sync"]["sigma"]["U"]
    sigma_Q_dust = params["dust"]["sigma"]["Q"]
    sigma_U_dust = params["dust"]["sigma"]["U"]
    sigma_templates = [sigma_Q_sync, sigma_U_sync, sigma_Q_dust, sigma_U_dust]
    return templates_map, scipy.linalg.block_diag(*[np.diag(s_) for s_ in sigma_templates])

def aggregate_mixing_params(params):
    temp_dust = params["dust"]["temp"]["mean"]
    beta_dust = params["dust"]["beta"]["mean"]
    beta_sync = params["sync"]["beta"]["mean"]
    templates_mixing = np.hstack([beta_dust, temp_dust, beta_sync])
    sigma_temp_dust = params["dust"]["temp"]["sigma"]
    sigma_beta_dust = params["dust"]["beta"]["sigma"]
    sigma_beta_sync = params["sync"]["beta"]["sigma"]
    sigma_templates_mixing = [sigma_beta_dust, sigma_temp_dust, sigma_beta_sync]
    return templates_mixing, scipy.linalg.block_diag(*[np.diag(s_) for s_ in sigma_templates_mixing])

def RBF_kernel(x, sigma = 1):
    return np.exp(-0.5*x/sigma)

def compute_discrepency_L2(tuple_input):
    ref_data, simulated_data = tuple_input
    return np.sum(np.abs(ref_data - simulated_data)**2)

def compute_discrepency_Inf(tuple_input):
    ref_data, simulated_data = tuple_input
    return np.max(ref_data - simulated_data)


def compute_acceptance_rates(discrepencies, epsilons, title, path):
    ratios = []
    for eps in epsilons:
        ratios.append(np.mean(np.random.binomial(1, RBF_kernel(np.array(discrepencies),eps))))

    plt.plot(epsilons, ratios)
    plt.title(title)
    plt.savefig(path)
    plt.close()

