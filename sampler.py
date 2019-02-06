import numpy as np
from fgbuster.mixingmatrix import MixingMatrix
from fgbuster.observation_helpers import get_instrument
import healpy as hp
from classy import Class
import pysm
from utils import get_pixels_params, get_mixing_matrix_params, aggregate_pixels_params, aggregate_mixing_params
from fgbuster.component_model import CMB, Dust, Synchrotron

COSMO_PARAMS_NAMES = ["n_s", "omega_b", "omega_cdm", "100*theta_s", "ln10^{10}A_s", "tau_reio"]
COSMO_PARAMS_MEANS = [0.9665, 0.02242, 0.11933, 1.04101, 3.047, 0.0561]
COSMO_PARAMS_SIGMA = [0.0038, 0.00014, 0.00091, 0.00029, 0.014, 0.0071]
L_MAX_SCALARS = 5000
LENSING = 'yes'
OUTPUT_CLASS = 'tCl pCl lCl'



class Sampler:
    def __init__(self, NSIDE):
        self.NSIDE = NSIDE
        self.Npix = 12*NSIDE**2
        print("Initialising sampler")
        self.cosmo = Class()
        print("Maps")
        self.templates_map, self.templates_var = aggregate_pixels_params(get_pixels_params(self.NSIDE))
        print("betas")
        self.matrix_mean, self.matrix_var = aggregate_mixing_params(get_mixing_matrix_params(self.NSIDE))
        print("Cosmo params")
        self.cosmo_means = np.array(COSMO_PARAMS_MEANS)
        self.cosmo_var = (np.diag(COSMO_PARAMS_SIGMA)/2)**2

        self.instrument = pysm.Instrument(get_instrument('litebird', self.NSIDE))
        self.components = [CMB(), Dust(150.), Synchrotron(150.)]
        self.mixing_matrix = MixingMatrix(*self.components)
        self.mixing_matrix_evaluator = self.mixing_matrix.evaluator(self.instrument.Frequencies)
        print("End of initialisation")

    def __getstate__(self):
        state_dict = self.__dict__.copy()
        del state_dict["mixing_matrix_evaluator"]
        del state_dict["cosmo"]
        del state_dict["mixing_matrix"]
        del state_dict["components"]
        return state_dict

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.cosmo = Class()
        self.components = [CMB(), Dust(150.), Synchrotron(150.)]
        self.mixing_matrix = MixingMatrix(*self.components)
        self.mixing_matrix_evaluator = self.mixing_matrix.evaluator(self.instrument.Frequencies)

    def sample_normal(self, mu, sigma, s = None):
        return np.random.multivariate_normal(mu, sigma, s)

    def sample_model_parameters(self):
        sampled_cosmo = self.sample_normal(self.cosmo_means, self.cosmo_var)
        #sampled_beta = self.sample_normal(self.matrix_mean, self.matrix_var).reshape((self.Npix, -1), order = "F")
        sampled_beta = self.matrix_mean.reshape((self.Npix, -1), order = "F")
        return sampled_cosmo, sampled_beta

    def sample_CMB_QU(self, cosmo_params):
        params = [('output', OUTPUT_CLASS),
                  ('l_max_scalars', L_MAX_SCALARS),
                  ('lensing', LENSING)]
        params += cosmo_params
        self.cosmo.set(params)
        self.cosmo.compute()
        cls = self.cosmo.lensed_cl(L_MAX_SCALARS)
        eb_tb = np.zeros(shape=cls["tt"].shape)
        print(cls["tt"].shape)
        _, Q, U = hp.synfast((cls['tt'], cls['ee'], cls['bb'], cls['te'], eb_tb, eb_tb), nside=self.NSIDE, new=True)
        self.cosmo.struct_cleanup()
        self.cosmo.empty()
        return Q, U

    def sample_mixing_matrix(self, betas):
        mat_pixels = []
        for i in range(self.Npix):
            m = self.mixing_matrix_evaluator(betas[i,:])
            mat_pixels.append(m)

        mixing_matrix = np.stack(mat_pixels, axis = 0)
        return mixing_matrix

    def sample_model(self):
        cosmo_params, sampled_beta = self.sample_model_parameters()
        maps = self.sample_normal(self.templates_map, self.templates_var)

        cosmo_dict = [(l[0],l[1]) for l in list(zip(COSMO_PARAMS_NAMES, cosmo_params.tolist()))]
        tuple_QU = self.sample_CMB_QU(cosmo_dict)
        map_CMB = np.stack(tuple_QU, axis = 1)
        mixing_matrix = self.sample_mixing_matrix(sampled_beta)
        map_Sync = np.stack([maps[0:self.Npix], maps[self.Npix:2*self.Npix]], axis = 1)
        map_Dust = np.stack([maps[2*self.Npix:3*self.Npix], maps[3*self.Npix:]], axis = 1)
        entire_map = np.stack([map_CMB, map_Dust, map_Sync], axis = 1)

        dot_prod = []
        for j in range(self.Npix):
            m = np.dot(mixing_matrix[j, :, :], entire_map[j, :, :])
            dot_prod.append(m)

        sky_map = np.stack(dot_prod, axis = 0)

        return {"sky_map": sky_map, "cosmo_params": cosmo_params, "betas": sampled_beta}






#sampler = Sampler(NSIDE)
#r = sampler.sample_model(1)
#['beta_d' 'temp' 'beta_pl']
#['beta_d' 'temp']
#['beta_pl']