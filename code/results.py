#%% 
import numpy as np
import scipy.optimize as opt
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
from implementation import *
import statsmodels.api as sm

XT, years_to_t, t_to_years, LATE_EXTREMES, LATE_YEARS = load_data()
BOUNDS = [(-0.5, 0.5)] + 2*[(-np.inf, np.inf)] + [(0., np.inf)] + [(-np.inf, np.inf)]
# XT['x'] has the yearly maxima
# XT['years'] is the years for the training data
# XT['t'] is the time variable (from 0 to 1 in training data)
# LATE_EXTREMES is the data for the last 10 years that exceed 140 cm
# LATE_YEARS is the years from 1940 to 2022

# %% 

def glm_plot(ax):
    years = t_to_years(XT['t'])
    x = sm.add_constant(years)
    y = XT['x']

    fam = sm.families.Gamma(link=sm.families.links.Log())
    model = sm.GLM(y, x, family=fam)
    result = model.fit()

    x_pred = LATE_YEARS
    x_pred_ = sm.add_constant(x_pred)
    mean_pred = result.get_prediction(x_pred_).summary_frame()['mean']
    cis = result.get_prediction(x_pred_).conf_int(alpha=0.05)

    ax.plot(years, y, label='Yearly maxima')
    l = ax.plot(x_pred, mean_pred, label='Gamma GLM')
    ax.fill_between(x_pred, cis[:, 0], cis[:, 1], alpha=0.2, color=l[0].get_color(), label='95% CI')
    ax.scatter(LATE_EXTREMES['year'], LATE_EXTREMES['sealevel'], color='black', label='Extreme events')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea level [cm]')
    return ax


def main_plot(LATE_EXTREMES, xi, mu0, mu1, sigma0, ax):
    years = t_to_years(XT['t'])
    x = sm.add_constant(years)
    y = XT['x']

    fam = sm.families.Gamma(link=sm.families.links.Log())
    model = sm.GLM(y, x, family=fam)
    result = model.fit()

    x_pred = np.arange(min(years), 2023)
    x_pred_ = sm.add_constant(x_pred)
    mean_pred = result.get_prediction(x_pred_).summary_frame()['mean']
    cis = result.get_prediction(x_pred_).conf_int(alpha=0.05)
    return_lvs_14, err_14 = return_level_k(xi, mu0, mu1, sigma0, 14, compute_err=True)
    return_lvs_30, err_30 = return_level_k(xi, mu0, mu1, sigma0, 30, compute_err=True)

    ax.plot(years, y, label='Yearly maxima')
    l = ax.plot(x_pred, mean_pred, label='Gamma GLM')
    ax.fill_between(x_pred, cis[:, 0], cis[:, 1], alpha=0.2, color=l[0].get_color(), label='95% CI')
    lbis_14 = ax.plot(LATE_YEARS, return_lvs_14, label=r'$R_{year,14}$')
    ax.fill_between(LATE_YEARS, return_lvs_14-err_14, return_lvs_14+err_14, alpha=0.2, color=lbis_14[0].get_color(), label='Delta 95% CI')
    lbis_30 = ax.plot(LATE_YEARS, return_lvs_30, label=r'$R_{year,30}$')
    ax.fill_between(LATE_YEARS, return_lvs_30-err_30, return_lvs_30+err_30, alpha=0.2, color=lbis_30[0].get_color(), label='Delta 95% CI')
    ax.scatter(LATE_EXTREMES['year'], LATE_EXTREMES['sealevel'], color='black', label='Extreme events')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('Year')
    ax.set_ylabel('Sea level [cm]')
    return ax

def qq_plot(x, xi, mus, sigma, ax):
    uniform_data = np.zeros(x.size)
    for i, mu in enumerate(mus):
        uniform_data[i] = stats.genextreme.cdf(x[i], c=xi, loc=mu, scale=sigma)

    ecdf = stats.ecdf(uniform_data)
    prob_low = np.concatenate(([0], ecdf.cdf.probabilities[:-1]))
    prob = 0.5 * ( prob_low + ecdf.cdf.probabilities ) 

    ax.scatter( ecdf.cdf.quantiles[:], prob, label='Data' )
    ax.plot([0, 1], [0,1], color='red', label='Identity line')
    ax.set_xlabel('Theoretical quantiles')
    ax.set_ylabel(r'$F^{-1}_{\xi,\mu(t),\sigma}(x_{t})$ quantiles')
    ax.legend()
    pass

def pdf_plot(data, xi, mus, sigma, ax):
    x = np.linspace(0.8*min(data), 1.1*max(data), 1000)
    avg_pdf = np.zeros(x.shape)
    for mu in mus:
        avg_pdf += stats.genextreme.pdf(x, c=xi, loc=mu, scale=sigma)
    avg_pdf /= len(mus)

    sns.kdeplot(data, label='Data', color='tab:blue', ax=ax)
    ax.plot(x, avg_pdf, label='GEV mixture', color='red')
    ax.set_xlabel('Sea level [cm]')
    ax.set_ylabel('Density')
    ax.legend()
    pass

def diagnostic_plots(x, xi, mus, sigma):
    fig, ax = plt.subplots( 1, 2, figsize=(12, 6), layout='tight' )
    qq_plot(x, xi, mus, sigma, ax[0])
    pdf_plot(x, xi, mus, sigma, ax[1])
    return

def return_level_k(xi, mu0, mu1, sigma0, k, compute_err=False):
    t_years = years_to_t(LATE_YEARS)
    mut = mu0 + mu1 * t_years
    retur = one_over_k_return_level(xi, mut, sigma0, k)
    err = None
    if compute_err:
        err = return_level_err(t_years, xi, mu0, mu1, sigma0, k)
    return retur, err

def bootstrap_ci( b, xi, mus, sigma, alpha=0.05 ):

    samples = np.zeros( (b, 2022-1940+1) )
    returns = np.zeros( (b, 2022-1940+1) )
    for i,mu in enumerate(mus):
        samples[:,i] = stats.genextreme.rvs(c=xi, loc=mu, scale=sigma, size=b)
    for j in range(b):
        xi0, mu0, mu1, sigma0, dummy = hard_guess_from_data('mu_rate',new_x=samples[j,:])
#        x0 = np.array([xi0, mu0, mu1, sigma0, dummy])
        x0 = np.array([-1.00000000e-01, 1.06061061e2, 1.77188579e1, 1.31667028e1,1.00000000e0])
        
        loss = lambda x: -time_dep_gev_ll('mu_rate', *x, new_x=samples[j,:])
        res_mu_rate = opt.minimize(loss, x0, method='SLSQP', bounds=BOUNDS)
        if not res_mu_rate.success:
            print('Optimization failed')
        xi, mu0, mu1, sigma0, dummy = res_mu_rate.x
        returns[j,:], _ = return_level_k(xi, mu0, mu1, sigma0, 14, compute_err=False)
        if j % 20 == 0:
            print(f'j={j}')
    
    returns = np.sort(returns, axis=0)
    #
    lower = int( (alpha/2) * b )
    upper = int( (1-alpha/2) * b )
    print(returns[0,:] - returns[-1,:])
    return returns[lower,:], returns[upper,:] 

# %%

fig, ax = plt.subplots( figsize=(10, 6), layout='tight' )
glm_plot(ax)
plt.savefig('../figures/glm-plot.png')

# %%
 
xi0, mu0, mu1, sigma0, sigma1 = hard_guess_from_data('sigma_rate')
x0 = np.array([xi0, mu0, mu1, sigma0, sigma1])
loss = lambda x: -time_dep_gev_ll('sigma_rate', *x)
res_sigma_rate = opt.minimize(loss, x0, method='SLSQP', bounds=BOUNDS, options={'disp': True})

ll_sigma_rate = -res_sigma_rate.fun

# %%

dummy = 1.0
xi0, mu0, mu1, sigma0, sigma1 = hard_guess_from_data('mu_rate')
x0 = np.array([xi0, mu0, mu1, sigma0, dummy])
loss = lambda x: -time_dep_gev_ll('mu_rate', *x)
res_mu_rate = opt.minimize(loss, x0, method='SLSQP', bounds=BOUNDS, options={'disp': True})
ll_mu_rate = -res_mu_rate.fun

xi, mu0, mu1, sigma0, dummy = res_mu_rate.x
mut = mu0 + mu1 * XT['t']
diagnostic_plots(XT['x'], xi, mut, sigma0)
plt.savefig('../figures/diagnostic-plot.png')

fig, ax = plt.subplots( figsize=(10, 6), layout='tight' )
ax = main_plot(LATE_EXTREMES, xi, mu0, mu1, sigma0, ax=ax)
plt.savefig('../figures/risk-analysis-plot.png')


'''
Bootstrap CI : does not work
b = 50
lower, upper = bootstrap_ci( b, xi, mut, sigma0, alpha=0.05 )
fig, ax = plt.subplots( 1, 1, figsize=(8, 6) )
ax.plot(LATE_YEARS, lower, color='tab:blue', label='bootstrap CI')
ax.plot(LATE_YEARS, upper, color='tab:blue')
ax.fill_between(LATE_YEARS, lower, upper, alpha=0.2, color='tab:blue', label='bootstrap CI')
'''

# %%

xi0, mu0, sigma0 = guess_from_data()
x0 = np.array([xi0, mu0, dummy, sigma0, dummy])
loss = lambda x: -time_dep_gev_ll('constant', *x)

res = opt.minimize(loss, x0, method='SLSQP', bounds=BOUNDS, options={'disp': True})
ll = -res.fun

# %%

p1 = chi2_test(ll_mu_rate, ll, 1)
print(f'Testing significance of time-varying mu w.r.t. constant model : p-value = {p1:.2e}')

p2 = chi2_test(ll_sigma_rate, ll_mu_rate, 1)
print(f'Testing significance of time-varying mu and sigma w.r.t. model w/out time-varying sigma : p-value = {p2:.2e}')

