#!/usr/bin/env python
# coding: utf-8

# # CERIDWEN
# ## from little tiny baby steps to stuff that could actually work for a paper in just 30 min! 
# 
# **Part A** we will do a lil test all together, with one model that I specified, so this will only encompass one model
# which we will fit to some some mock photometry. 
# 
# 
# But you probably already have a specific science case 
# in your head which just can not wait!
# So **Part B** will be a little bit more free, where
#  you can choose the kind of observation
# and populate it with real data. Then you can choose
#  the libraries you want to use 
# the *grid* (SSPData), frankenstein a forward model,
#  fit it and post-process it.
# Maybe this will happen fast enough for someone to
#  present at the end! (So lock in please.)
# 
# 
# Everything runs on the GPU (Apple GPU in float32, or NVIDIA in float64).

# In[ ]:


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp

import ceridwen
from ceridwen import (SSPData, CSPBasis, SedModel, fitSED, PostProcess, Cosmology,
                      Kinematics, Instrument, TIED, load_postprocess)
from ceridwen.observation import Photometry, Spectrum, Lines
from ceridwen.observation.filters import list_available_filters
from ceridwen.model import logsfr_ratios_to_sfh
from ceridwen.priors import Uniform, ClippedNormal, StudentT, LogUniform
from ceridwen.ssps import fetch_grid, available_grids
from ceridwen.precision import precision_name
from ceridwen.plotting import summary_figure, corner_figure

print("ceridwen  :", ceridwen.__version__)
print("device    :", jax.devices())
print("precision :", precision_name())
print("SPS_HOME  :", os.environ.get("SPS_HOME"))
_v = tuple(int(x) for x in ceridwen.__version__.split(".")[:3])
if _v < (1, 0, 14):
    raise RuntimeError(f"this notebook needs ceridwen >= 1.0.14 (model.display() with emission "
                       f"lines, CSPBasis(fixed=...)), found {ceridwen.__version__}: "
                       f"pip install -U ceridwen")
if jax.default_backend() == "cpu":
    raise RuntimeError("JAX runs on the CPU; this tutorial needs the GPU (see the prep email)")

NJY = 3631e9                      # nJy per AB maggie


# # Part A: the simplest fit
# We fill fit in 5 steps. We will choose the stellar libraries we use, the model, make the mock data, fit it and plot the results. 
# The galaxy is a mock at z = 0.1 observed in twelve bands from GALEX to WISE, made with the same model, so we know the answer.
# (and do not have to worry about mode misspecification, which you should all remember from my brilliant Journal Club presentation yesterday)

# ### A1. The stellar population grid
# 
# `fetch_grid` downloads a published grid and returns its path so you do not have to download it next time you use this exact code. We use it to build our SSP = Single Stellar Population object, which is basically a grid of stellar spectra over age and metallicity.
# `display()` prints the details you should check before deciding if this is the right grid for you. It is also a great function in case you forgot which one you loaded.
# 

# In[ ]:


ssp = SSPData.load(fetch_grid("mist_miles_chab"))
#ssp.display()


# ### A2. The model
# 
# Next we need to decide which of the physics are relevant for our problem and which of them we want to include/exclude. For that we build a 
# `CSPBasis`, CSP = Composite Stellar Population, and turn possible ingredients on or off. If we want to fit for something it needs 
# to be included here.
# 
# `lookback_time` holds the SFH nodes in Gyr, index 0 = today. Here: a 6-bin step SFH (constant SFR within each bin, 7 nodes), one metallicity, diffuse dust and nebular emission (continuum and lines,
# from CLOUDY grids in `$SPS_HOME`).
# 
# Switching the nebular model on adds two parameters to the CSP: the gas-phase metallicity
# `gas_logz` and the ionisation parameter `gas_logu` (both log10). We fit them, so they get a prior
# below, like every other free parameter. (To keep the lines but not fit these two, fix them when
# you build the CSP: `CSPBasis(..., fixed={"gas_logz": -0.3, "gas_logu": -2.5})`. A fixed parameter
# needs no prior.)

# In[ ]:


cosmo = Cosmology.planck18()
Z_A = 0.1
lookback_A = [0.0, 0.01, 0.03, 0.1, 0.5, 2.0, float(cosmo.age(Z_A))]

csp_A = CSPBasis(ssp, lookback_time=jnp.asarray(lookback_A), cosmo=cosmo,
                 zh_const=True, sfh_interp="step",
                 add_dust=False, add_diffuse_dust=True, add_neb=True,
                 verbose=False)
print("CSP parameters:", csp_A.param_names)



# In[ ]:


csp_A


# We do not want to fit all of these parameters directly, since it can be very hard to define priors for something like an absolute star formation rate, so instead we define 'transforms'. These take basically anything as an input, and are basically just a reparametrisation. In this example, we want to fit for the ratio between sfh bins, so we need to write a function that translates them back to the SFRs that the CSPBasis expects:

# In[ ]:


times_A = np.array(csp_A.sfh_times)
TRANSFORMS_A = {"sfh": lambda th, _t=times_A: logsfr_ratios_to_sfh(th["logsfr_ratios"], sfh_times_yr=_t)}


# The model also includes the priors which we use for the fit. Each free parameter needs a prior:

# In[ ]:


PRIORS_A = {"logsfr_ratios": StudentT(mean=0.0, scale=0.3, df=2.0),
            "logmass": Uniform(low=8.0, high=12.0),
            "logzsol": Uniform(low=-2.0, high=0.2),
            "diffuse_tau_kc": ClippedNormal(mean=0.3, sigma=1.0, low=0.0, high=4.0),
            "diffuse_dust_index": Uniform(low=-1.0, high=0.4),
            "gas_logz": Uniform(low=-1.5, high=0.2),       # gas-phase log(Z/Z_sun)
            "gas_logu": Uniform(low=-3.5, high=-1.0)}      # log ionisation parameter


# And because it is a mock test, we also define the truths:

# In[ ]:


TRUTH_A = {"logsfr_ratios": jnp.array([0.2, 0.3, 0.1, -0.3, -0.5]), "logmass": jnp.array([10.3]),
           "logzsol": jnp.array([-0.3]), "diffuse_tau_kc": jnp.array([0.6]),
           "diffuse_dust_index": jnp.array([-0.5]),
           "gas_logz": jnp.array([-0.3]), "gas_logu": jnp.array([-2.5])}


# ### A3. The data
# Next we are generating the data:
# Photometry is in AB maggies (1 maggie = 3631 Jy). We push the truth through the model and add
# noise at S/N = 20.
# 
# We also measure five emission lines. Line fluxes are integrated, observed-frame fluxes in
# erg s$^{-1}$ cm$^{-2}$. CERIDWEN identifies a
# line by its row in the nebular grid and its vacuum rest wavelength; `csp_A.neb.nebem_line_pos` and
# `csp_A.neb.nebem_line_names` list them.

# In[ ]:


FILTERS_A = ["galex_FUV", "galex_NUV", "sdss_u0", "sdss_g0", "sdss_r0", "sdss_i0", "sdss_z0",
             "twomass_J", "twomass_H", "twomass_Ks", "wise_w1", "wise_w2"]
n_A = len(FILTERS_A)

phot = Photometry(filters=FILTERS_A, 
           flux=np.zeros(n_A),
           uncertainty=np.ones(n_A), 
           name="phot")

# five lines by their vacuum rest wavelength [A]: the nearest row of the nebular grid
LINES_WANT = {"H beta": 4862.7, "[O III] 5007": 5008.2, "H alpha": 6564.6,
              "[N II] 6584": 6585.3, "[S II] 6717": 6718.3}



LINES_WANT = {"H beta": 4862.7, "[O III] 5007": 5008.2, "H alpha": 6564.6,
              "[N II] 6584": 6585.3, "[S II] 6717": 6718.3}     # vacuum rest wavelengths [A]
pos = np.asarray(csp_A.neb.nebem_line_pos)
ROWS_A = [int(np.argmin(np.abs(pos - w))) for w in LINES_WANT.values()]   # ceridwen re-checks these
n_L = len(ROWS_A)

def make_lines_A(flux, err):
    return Lines(line_ind=ROWS_A, line_names=list(LINES_WANT), wavelength=[float(pos[r]) for r in ROWS_A],
                 flux=np.asarray(flux), uncertainty=np.asarray(err), name="lines")


lines = make_lines_A(np.zeros(n_L), np.ones(n_L))

obs = [phot, lines]

#phot.display()


# In[ ]:


csp_A


# In[ ]:


#lines.display()


# The photometry and the lines are still empty at the moment, but we only need the filter information that defines how the photometry should look like for the next step. As soon as we have the observation strucuture, we have all three ingredients for the SedModel, which takes the physical model (CSPBasis) and returns to us what we would expect the observation(s) to look like.

# In[ ]:


truth_model_A = SedModel(csp_A,
                         obs,
                         priors=PRIORS_A, 
                         transforms=TRANSFORMS_A, 
                         free_param_init=TRUTH_A, 
                         zred=Z_A)


# now we can use the SED model to predict the mock data by running SedModel.predict() while giving it the parameter vector we defined earlier, TRUTH_A

# In[ ]:


pred_A = truth_model_A.predict(TRUTH_A)
true_A = np.asarray(pred_A["phot"])
rng = np.random.default_rng(1)
err_A = true_A / 20.0

phot_A = Photometry(filters=FILTERS_A, flux=true_A + err_A * rng.standard_normal(n_A),
                    uncertainty=err_A, name="phot")

true_L = np.asarray(pred_A["lines"])           # erg/s/cm^2
err_L = true_L / 10.0                          # S/N = 10 per line
lines_A = make_lines_A(true_L + err_L * rng.standard_normal(n_L), err_L)
for label, f, e in zip(LINES_WANT, true_L, err_L):
    print(f"  {label:14s} {f:.3e} +- {e:.1e} erg/s/cm^2")


# Now we can build a new model, which we give the now filled photometry and line objects (the rest does not change.) Also we need to initialise all free parameters that the csp does not know already, so it knows which shape the inputs will have when a transform happens. (`logmass` needs only its prior: ceridwen starts it at the prior's median.) Then we can look at the now complete model.

# In[ ]:


model_A = SedModel(csp_A, [phot_A, lines_A], priors=PRIORS_A, transforms=TRANSFORMS_A,
                   free_param_init={"logsfr_ratios": jnp.zeros(5)},
                   zred=Z_A)
#model_A.display()


# ### A4. The fit
# 
# Nested sampling. `num_live` live points; `num_delete` are replaced per step, in parallel on
# the GPU; `num_inner_steps` slice steps per replacement. These are demo settings.

# In[ ]:


result_A = fitSED(model_A, sampler="nested", rng_key=jax.random.PRNGKey(0),
                  sampler_kwargs={"num_live": 150, "num_delete": 30, "num_inner_steps": 30}, #for a proper result: sampler_kwargs={"num_live": 500, "num_delete": 100, "num_inner_steps": 60, "logZ_tol": -5.0},
                  output_dir="fit_partA")
print(f"ln Z = {result_A.log_evidence:.2f} +- {result_A.log_evidence_err:.2f}, "
      f"{result_A.wall_time_s:.0f} s")


# ### A5. The result
# 
# `PostProcess` turns the weighted samples into equal-weight draws and pushes each through the
# model: parameters, derived quantities, posterior-predictive data.

# In[ ]:


out_A = PostProcess(model_A, result_A, n_samples=1000).run()
truths_A = {k: (float(v[0]) if v.size == 1 else np.asarray(v)) for k, v in TRUTH_A.items()}


def crop_sed(fig, model):
    # summary_figure pads the data range by 25 % in log wavelength; crop the SED panel (and the
    # residuals that share its x axis) to the filter edges and the spectrum ends [micron]
    lo, hi = [], []
    for o in model.observations:
        if isinstance(o, Photometry):
            for f in o.filters:
                w = f.wavelength[f.transmission > 0.01 * f.transmission.max()]
                lo.append(w.min()); hi.append(w.max())
        elif isinstance(o, Spectrum):
            lo.append(float(np.min(o.wavelength))); hi.append(float(np.max(o.wavelength)))
    if lo:
        fig.axes[0].set_xlim(min(lo) / 1e4, max(hi) / 1e4)
    return fig


crop_sed(summary_figure(out_A, model_A, title="Part A: mock galaxy", truths=truths_A), model_A)
#plt.show()
plt.savefig('test.png',bbox_inches='tight')

