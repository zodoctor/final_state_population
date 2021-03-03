#!/usr/bin/env python
# coding: utf-8

# ./Final_State_Population.py --n_event_samples 20000 --f_ref 20 --regen --n_cpu 4

# # Characterize the population of remnant black holes

# In[1]:

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--regen',default=False,action='store_true')
parser.add_argument('--n_event_samples',default=20000,type=int)
parser.add_argument('--n_cpu',default=1,type=int)
parser.add_argument('--f_ref',default=None,type=float)
args = parser.parse_args()

import os
import numpy as np
np.random.seed(1234)
import numpy.lib.recfunctions
import h5py
import json
from matplotlib import pyplot as plt
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from scipy.interpolate import interp1d
import matplotlib 
matplotlib.rcParams['text.usetex'] = True
from scipy.stats import gaussian_kde
import gwpopulation as gwp
import surfinBH
from multiprocessing import Pool
import matplotlib.transforms as mtransforms
import collections.abc
import copy


# ### Now integrate the rate over time. 
# We'll assume the rate follows the cosmic star formation rate, but with some fiducial delay time distribution.  We'll use the star formation rate from [Madau and Fragos](https://arxiv.org/pdf/1606.07887.pdf):
# $$\psi(z) = 0.01 \frac{(1+z)^{2.6}}{1+[(1+z)/3.2]^{6.2}} \textrm{M}_\odot \textrm{yr}^{-1}\textrm{Mpc}^{-3}$$
# For the delay time distribution, we'll assume:
# $$p(t_d) \propto t_d^{-1}\ \textrm{if}\ t_d>10\ \textrm{Myr, else}\ 0$$
# The merger rate over cosmic time is:
# $$R(t) = A \int_0^{t} p(t-t')\psi(t') dt'$$
# And we know that $R(t_h) = R_0$ where $R_0$ is the present-day rate inferred by the LVC, so:
# $$A = \frac{R_0}{\int_0^{t_h} p(t-t')\psi(t') dt'}$$
# The current number density of remnants is:
# $$n = \int_0^{t_h} R(t) dt = R_0\frac{\int_0^{t_h} \int_0^{t} p(t-t')\psi(t') dt'}{\int_0^{t_h} p(t-t')\psi(t') dt'}$$

# In[2]:


class RateCalculator(FlatLambdaCDM):
    def __init__(self,H0=70., Om0=0.3,alpha=1,td_min=1e7):
        self.dist_unit = u.megaparsec
        self.time_unit = u.year
        self.alpha= alpha
        self.td_min = td_min #minimum delay time
        super().__init__(H0,Om0)
        self.z_min,self.z_max = 0.0001,15
        #self.zgrid = np.linspace(self.z_min,self.z_max,2000)
        self.zgrid = np.logspace(np.log10(self.z_min),np.log10(self.z_max),5000)
        self.t_of_zgrid = self.hubble_time.to_value(self.time_unit) - self.lookback_time(self.zgrid).to_value(self.time_unit)
        self.t_min = np.amin(self.t_of_zgrid) # age of universe at z_max
        self.t_of_z = interp1d(self.zgrid,self.t_of_zgrid)
        self.z_of_t = interp1d(self.t_of_zgrid,self.zgrid)
        self.R_norm = self.R_of_t(self.hubble_time.to_value(self.time_unit))
        
        
    def sfr_of_t(self,t):
        """Return star formation rate in units of solar masses per year per cubic Mpc"""
        z = self.z_of_t(t) 
        return self.sfr_of_z(z)
        
    def sfr_of_z(self,z):
        return 0.01*(1.+z)**2.6 / (1.+((1.+z)/3.2)**6.2)

    def p_delay_time(self,t):
        """un-normalized probability of a given delay time for a 
        delay time distribution t_d^-alpha where t_d > t_min, else 0
        """
        return np.array([ti**(-self.alpha) if ti>self.td_min else 0 for ti in t])
    
    def R_of_t(self,t):
        sel = self.t_of_zgrid <= t
        # negative sign below to deal with t_of_zgrid going from high to low
        return -np.trapz(self.sfr_of_t(self.t_of_zgrid[sel])*self.p_delay_time(t-self.t_of_zgrid[sel]),self.t_of_zgrid[sel])
    
    def number_density(self,t=None,present_rate=1e-7):
        if t is None:
            t = self.hubble_time.to_value(self.time_unit)
        integrand_vals = np.array(list(map(self.R_of_t,self.t_of_zgrid)))
        sel = self.t_of_zgrid <= t
        return -present_rate*np.trapz(integrand_vals[sel],self.t_of_zgrid[sel])/self.R_norm
        

rate_calc = RateCalculator()
number_density = rate_calc.number_density()
print(f'the current number density of remnant black holes is {number_density} per cubic megaparsec')


# ## Re-vamp with O3a 
# Here we're going to use the latest and greatest from O3a results.  The power-law + peak model had the highest BF in the O3a populations paper, so that's the one we'll use for this analysis.  The tricky part is pulling individual merger samples for each hyperparameter sample.  We'll do that using a mix of reverse CDF sampling and rejection sampling


# Get population hyperparameter samples for the power law + peak samples
# Load Power Law + Peak samples
with open("Population_Samples/default/o1o2o3_mass_c_iid_mag_two_comp_iid_tilt_powerlaw_redshift_result.json","r") as jfile:
    jf = json.load(jfile)
    print(jf['priors'])
    posterior_samples = jf['posterior']['content']
    
# add in the mean and variance of the spin mag beta distribution converted to alpha and beta:
mu = np.array(posterior_samples['mu_chi'])
sigma_sqrd = np.array(posterior_samples['sigma_chi']) # this is mislabeled in the file as sigma instead of variance
posterior_samples['alpha_chi'] = ((1. - mu) / sigma_sqrd - 1. /mu ) * (mu ** 2.)
posterior_samples['beta_chi'] = posterior_samples['alpha_chi']*(1.-mu)/mu


np.median(posterior_samples['log_10_rate'])


# Set up the grids which will be sampled based on population weighting


mass_dist_params = ['alpha', 'beta', 'mmax', 'mmin', 'lam', 'mpp', 'sigpp', 'delta_m']
spinmag_dist_params = ['alpha_chi', 'beta_chi']
spintilt_params = ['xi_spin', 'sigma_spin']
hyp_name_dict = {'mass':mass_dist_params,
               'spinmag':spinmag_dist_params,
               'spintilt':spintilt_params
              }

masspop = gwp.models.mass.SinglePeakSmoothedMassDistribution()
spinmagpop = gwp.models.spin.iid_spin_magnitude_beta #(dataset, xi_spin, sigma_spin, amax, alpha_chi, beta_chi)
spintiltpop = gwp.models.spin.iid_spin_orientation_gaussian_isotropic
pops = {'mass':masspop,'spinmag':spinmagpop,'spintilt':spintiltpop}

# set up mass grid 
m1s_grid = np.linspace(3,100,400)
qs_grid = np.linspace(0.01,1,100)
M1s,Qs = np.meshgrid(m1s_grid,qs_grid)
mass_pts = np.array(list(zip(M1s.flatten(),Qs.flatten())),dtype=[('mass_1',float),('mass_ratio',float)])

# set up spin mag grid
as_grid = np.linspace(0,1,100)
A1s,A2s = np.meshgrid(as_grid,as_grid)
as_pts = np.array(list(zip(A1s.flatten(),A2s.flatten())),dtype=[('a_1',float),('a_2',float)])

# set up spin tilt grid
costilts_grid = np.linspace(-1,1,100)
Ctilts1,Ctilts2 = np.meshgrid(costilts_grid,costilts_grid)
costilts_pts = np.array(list(zip(Ctilts1.flatten(),Ctilts2.flatten())),dtype=[('cos_tilt_1',float),('cos_tilt_2',float)])

grids = {'mass':mass_pts,'spinmag':as_pts,'spintilt':costilts_pts}

def get_event_samples(hypersample,n_samples=1000):
    sample_arr = []
    for hypname in pops.keys():
        hyp_dict = {param:hypersample[param] for param in hyp_name_dict[hypname]}
        ps = pops[hypname](grids[hypname],**hyp_dict)
        samples = np.random.choice(grids[hypname],size=n_samples,replace=True,p=ps/np.sum(ps))
        sample_arr.append(samples)
    return np.lib.recfunctions.merge_arrays(sample_arr,flatten=True,usemask=False)
    
def mag_tilt_to_components(a,cos_tilt):
    phase = np.random.uniform(2*np.pi)
    sin_tilt = np.sqrt(1.-cos_tilt**2)
    return [
        a*sin_tilt*np.cos(phase), #x
        a*sin_tilt*np.sin(phase), #y
        a*cos_tilt #z
    ]    

def get_weights(hypersample,event_samples):
    ps_arr = []
    for hypname in pops.keys():
        hyp_dict = {param:hypersample[param] for param in hyp_name_dict[hypname]}
        ps_arr.append(pops[hypname](event_samples,**hyp_dict))
    return np.multiply.reduce(ps_arr)


# Get mass and spin samples from the maxL population
max_L_idx = np.argmax(posterior_samples['log_likelihood'])

# In[7]:

if args.regen or not os.path.exists('final_params.npy'):

    n_event_samples = args.n_event_samples
    fid_pop_samples = get_event_samples({key:value[max_L_idx] for key,value in posterior_samples.items()},
                                        n_samples=n_event_samples)


    # Now use SurfinBH to get the final masses and spins

    fit7dq4 = surfinBH.LoadFits('NRSur7dq4Remnant')
    fit3dq8 = surfinBH.LoadFits('NRSur3dq8Remnant')

    G = 6.67e-11 # Units: m^3 / (kg s^2)
    c = 3e8 # Units: m/s
    Msol = 1.989e30 # Units: kg

    def calc_final_state(samples):
        q = 1./samples['mass_ratio'] # use q>=1 definition
        if q <= 6:
            chi_1 = mag_tilt_to_components(samples['a_1'],samples['cos_tilt_1'])
            chi_2 = mag_tilt_to_components(samples['a_2'],samples['cos_tilt_2'])
            mtot = Msol*samples['mass_1']*(1.+1./q) # Units: kg
            if args.f_ref:
                omega0 = 2.*np.pi*(args.f_ref/2.)*G*mtot/(c**3)
                mf, chif, vf, _, _, _ = fit7dq4.all(q, chi_1, chi_2,omega0=omega0,allow_extrap=True,PN_dt=5)
            else:
                mf, chif, vf, _, _, _ = fit7dq4.all(q, chi_1, chi_2,allow_extrap=True)
            return mf*samples['mass_1']*(1.+1./q),np.linalg.norm(chif),np.linalg.norm(vf)
        else:
            chi_1 = [0,0,samples['a_1']*samples['cos_tilt_1']]
            chi_2 = [0,0,samples['a_2']*samples['cos_tilt_2']]
            mf, chif, vf, _, _, _ = fit3dq8.all(q, chi_1, chi_2,allow_extrap=True)
            return mf*samples['mass_1']*(1.+1./q),np.linalg.norm(chif),np.linalg.norm(vf)


    import warnings
    warnings.filterwarnings("ignore", category=UserWarning) 
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=h5py.h5py_warnings.H5pyDeprecationWarning)
    if args.n_cpu>1:
        pool = Pool(args.n_cpu)
        final_params = np.array(pool.map(calc_final_state,fid_pop_samples))
    else:
        final_params = []
        for i in range(len(fid_pop_samples)):
            print(f'finished processing {i}/{args.n_event_samples}')
            final_params.append(calc_final_state(fid_pop_samples[i]))
        final_params = np.array(final_params)
    np.save('final_params.npy',final_params)
    np.save('fid_pop_samples.npy',fid_pop_samples)
else:
    final_params,fid_pop_samples = np.load('final_params.npy'),np.load('fid_pop_samples.npy')
    n_event_samples = len(fid_pop_samples)

print('got remnant params for fiducial samples')

def weighted_percentile(a,w,q):
    """
    a: array of samples
    w: weights of samples
    q: percentiles
    """
    w=w/np.sum(w)
    sorta = np.array([s for s,_ in sorted(zip(a,w))])
    sortw = np.array([wt for _,wt in sorted(zip(a,w))])
    if isinstance(q,collections.abc.Iterable):
        return np.array([sorta[np.cumsum(sortw)>(qi/100.)][0] for qi in q])
    else:
        return sorta[np.cumsum(sortw)>(q/100.)][0]

n_grid_pts = 100 
mass_plot_grid = np.linspace(6,200,n_grid_pts)
chif_plot_grid = np.linspace(0,1,n_grid_pts)
vf_plot_grid = np.linspace(0,4,n_grid_pts)
#scales = [1.,1.,3e5]
transforms = [lambda x:x, lambda x:x, lambda x:(np.log10(3e5*x))]
plot_grids = [mass_plot_grid,chif_plot_grid,vf_plot_grid]
xlabels = ['M_f','\chi_f','v_f']

base_rate = rate_calc.number_density(present_rate=1.)

fid_weights = get_weights({key:value[max_L_idx] for key,value in posterior_samples.items()},fid_pop_samples)

fig,ax = plt.subplots(1,3,figsize=(14,5))

n_hyper_lines = 500 
n_params = len(xlabels)
percentiles = [1,50,99]
pctl_array = np.empty((len(percentiles),n_hyper_lines,n_params))
lines = np.empty([n_grid_pts,n_hyper_lines,n_params])
rates = []
hyp_idx = []
wts_arr = []
for i,k in enumerate(np.random.choice(len(posterior_samples['alpha']),size=n_hyper_lines,replace=False)):
    hyp_idx.append(k)
    wts = get_weights({key:value[i] for key,value in posterior_samples.items()},fid_pop_samples)/fid_weights ;
    wts[np.isnan(wts) | np.isinf(wts)] = 0.
    wts = wts/np.sum(wts)
    wts_arr.append(wts)
    rate = base_rate*(10.**posterior_samples['log_10_rate'][i])*1e-9 #1e-9 to convert to Mpc^3
    rates.append(rate)
    for j in range(len(ax)):
        pctl_array[:,i,j] = weighted_percentile(transforms[j](final_params[:,j]),wts,q=percentiles)
        lines[:,i,j] = rate*gaussian_kde(transforms[j](final_params[:,j]),weights=wts)(plot_grids[j])

hyp_idx = np.array(hyp_idx)
wts_arr = np.array(wts_arr)
        
rates = np.array(rates)
m_rate,l_rate,u_rate = np.median(rates),np.percentile(rates,5),np.percentile(rates,95)
rates_mwev = rates*10. # 0.1 galaxy per cubic megaparsec -> 10 cubic megaparsecs per galaxy
m_rate_mwev,l_rate_mwev,u_rate_mwev = np.median(rates_mwev),np.percentile(rates_mwev,5),np.percentile(rates_mwev,95)


# In[242]:


print(f'the number density of remnant black holes is {m_rate:2.1f} (-{m_rate-l_rate:2.1f}/+{u_rate-m_rate:2.1f}) per cubic Mpc')
print(f'the number density of remnant black holes is {m_rate_mwev:2.1f} (-{m_rate_mwev-l_rate_mwev:2.1f}/+{u_rate_mwev-m_rate_mwev:2.1f}) per MW ev galaxy')


# In[320]:


units = ['[\mathrm{{M}}_{{\odot}}]','','\;[\mathrm{km/s}]']
yunits = ['\;[\mathrm{{M}}_\odot^{{-1}}\mathrm{{Mpc}}^{{-3}}]','\;[\mathrm{{Mpc}}^{{-3}}]','\;[\mathrm{{Mpc}}^{{-3}}]']
logunits = [False,False,True]
xlims = [(0,200),(0,1),(0,4)]
ylims = [(1e-2,1e2),(1e1,1e5),(1e0,1e4)]
xlabels = ['M_f','\chi_f','v_f']

for j in range(len(ax)):
    ax[j].tick_params(labelsize=13)
    paramlines = lines[:,:,j]
        
    # plot the individual population lines
    for i in range(lines.shape[1]):
        ax[j].plot(plot_grids[j],paramlines[:,i],color='k',alpha=0.5,zorder=0,lw=0.1)
    
    # plot the credible region and median
    ax[j].fill_between(plot_grids[j],np.percentile(paramlines,5,axis=1),np.percentile(paramlines,95,axis=1),color='b',alpha=0.3)
    ax[j].plot(plot_grids[j],np.median(paramlines,axis=1),color='r',linestyle='--',lw=3)
    ax[j].set_ylim(ylims[j])
    ax[j].set_xlim(xlims[j])
    ax[j].set_yscale('log')
    
    # If using a pre-logged variable
    if logunits[j]:
        ax[j].set_xlabel(r'$\log_{{10}}({}{})$'.format(xlabels[j],units[j]),fontsize=16)
        ax[j].set_ylabel(r'$\frac{{dN}}{{d(\log_{{10}}{})dV}}{}$'.format(xlabels[j],yunits[j]),fontsize=16)
    else:
        ax[j].set_xlabel(f'${xlabels[j]}{units[j]}$',fontsize=16)
        ax[j].set_ylabel(r'$\frac{{dN}}{{d{}dV}}{}$'.format(xlabels[j],yunits[j]),fontsize=16)
                           
    # plot the percentile credible regions
    trans = mtransforms.blended_transform_factory(ax[j].transData, ax[j].transAxes)
    for k in [0,2]:
        ax[j].fill_between(
            np.array([np.percentile(pctl_array[k,:,j],10),np.percentile(pctl_array[k,:,j],90)]),
            0,1,
            facecolor='gray',alpha=0.3,transform=trans)

ax[2].annotate("", xy=(0.25, 0.95), xytext=(0.39/4, 0.95), arrowprops=dict(arrowstyle="<->"),xycoords='axes fraction')
ax[2].annotate(r"$\mathrm{YSC}$", xy=(0.13, 0.96),xycoords='axes fraction')

ax[2].annotate("", xy=(1.7/4, 0.95), xytext=(0.25, 0.95), arrowprops=dict(arrowstyle="<->"),xycoords='axes fraction')
ax[2].annotate(r"$\mathrm{GC}$", xy=(0.315, 0.96),xycoords='axes fraction')

ax[2].annotate("", xy=(2.4/4, 0.95), xytext=(1.7/4, 0.95), arrowprops=dict(arrowstyle="<->"),xycoords='axes fraction')
ax[2].annotate(r"$\mathrm{NSC}$", xy=(0.48, 0.96),xycoords='axes fraction')


fig.tight_layout()
fig.savefig('final_param_dist.pdf')
fig.savefig('final_param_dist.png',dpi=300)


# In[334]:


new_wts_arr = np.empty(n_hyper_lines*n_event_samples)
new_fin_params_arr = np.empty((n_hyper_lines*n_event_samples,final_params.shape[1]))
for i in range(n_hyper_lines):
    new_wts_arr[(i*n_event_samples):((i+1)*(n_event_samples))] = wts_arr[i,:]
    new_fin_params_arr[i*n_event_samples:(i+1)*(n_event_samples),:] = final_params

transforms = [lambda x:np.log10(x), lambda x:x, lambda x:(np.log10(3e5*x))]
labels = [r'\log_{10}(M_f\; [M_{\odot}])',r'\chi_f',r'\log_{10}(v_f\; [\mathrm{km/s}])']
names = ['Mf','xf','vf']
fig,ax = plt.subplots(1,3,figsize=(15,5))
ax = ax.flatten()
ax_num=0
for j in range(n_params):
    for k in range(n_params):
        if k <= j:
            continue    
        #fig,ax = plt.subplots()
        my_cmap = plt.cm.viridis
        ax[ax_num].set_facecolor(my_cmap(0))
        im = ax[ax_num].hist2d(transforms[j](new_fin_params_arr[:,j]),transforms[k](new_fin_params_arr[:,k]),
                   weights=new_wts_arr,density=True,bins=25,norm=matplotlib.colors.LogNorm(),cmap=my_cmap)
        ax[ax_num].set_xlabel(r'$'+labels[j]+r'$',fontsize=16)
        ax[ax_num].set_ylabel(r'$'+labels[k]+r'$',fontsize=16)
        ax[ax_num].set_title(r'$p\left('+labels[j]+','+labels[k]+r'\right)$',fontsize=16)
        fig.colorbar(im[3], ax=ax[ax_num])
        ax_num += 1
fig.tight_layout()
fig.savefig(f'remnant_2d_dist.pdf')
fig.savefig(f'remnant_2d_dist.png',dpi=300)





