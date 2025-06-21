import numpy as np
from scipy import stats 
from statsmodels.tools.numdiff import approx_hess1
import pandas as pd
import matplotlib.pyplot as plt

## increasing font sizes for the figures
plt.rc( 'axes', titlesize=17 ) 
plt.rc( 'axes', labelsize=15 ) 
#plt.rc( 'lines', linewidth=2.2 ) 
plt.rc( 'xtick', labelsize=12 ) 
plt.rc( 'ytick', labelsize=12 )
plt.rc( 'legend',fontsize=12 )

def load_data():
    extremes = pd.read_csv('../data/extreme.csv')
    extremes = extremes[extremes['year']>=2013]
    relevant = extremes[extremes['sealevel']>140]
    full_years = np.arange(1940, max(relevant['year'])+1)

    data = pd.read_csv('../data/venice90.csv')
    maxima = data.groupby('year')['sealevel'].max()

    xt = {}
    xt['x'] = maxima.values
    xt['years'] = maxima.index
    years_to_t = lambda years: (years - xt['years'].min()) / \
                            (xt['years'].max() - xt['years'].min())
    t_to_years = lambda t: t * (xt['years'].max() - xt['years'].min()) \
                                    + xt['years'].min()
    xt['t'] = years_to_t(xt['years'])
    return xt, years_to_t, t_to_years, relevant, full_years

XT, _, __, ___, ____ = load_data()

def t(xi, mu, sigma, new_x=None):
    data = XT['x']
    if new_x is not None:
        data = new_x
    result = np.zeros(data.shape)
    is_def = 1 + xi * (data - mu) / sigma > 0
    ok_data = data[is_def]
    if xi == 0:
        result[is_def] = np.exp( - (ok_data - mu) / sigma )
    else:
        result[is_def] = (1 + xi * (ok_data - mu) / sigma) ** (-1 / xi)
    result[~is_def] = np.nextafter(0, 1)
    return result

def time_dep_gev_ll(model_id : str, xi, mu0, mu1, sigma0, sigma1=None, new_x=None):
    data = XT
    if model_id != 'no':
        if model_id != 'sigma_rate':
            sigma1 = 0.0
        if model_id == 'constant':
            mu1 = 0.0
    if new_x is not None:
        data['x'] = new_x
    t_mu = mu0 + mu1 * data['t']
    t_sigma = sigma0 + sigma1 * data['t']
    return np.sum(
        - np.log(t_sigma)
        + (1 + xi) * np.log( t(xi, t_mu, t_sigma, data['x']) )
        - t(xi, t_mu, t_sigma, data['x'])
    )

# log likelihood function for GEV modeling 
def gev_log_likelihood(xi, mu, sigma):
    return time_dep_gev_ll('no', xi, mu, 0.0, sigma, 0.0)

def guess_from_data():
    sigma = np.std(XT['x']) * np.sqrt(6) / np.pi
    mu = np.mean(XT['x']) - np.euler_gamma * sigma
    xi = -0.1
    return xi, mu, sigma

def hard_guess_from_data(model_id: str,new_x=None):
    x = XT['x']
    if new_x is not None:
        x = new_x
    if model_id == 'sigma_rate':
        first_third_data = x[:len(x)//3]
        sigma0 = np.std(first_third_data) * np.sqrt(6) / np.pi
        mu0 = np.mean(first_third_data) - np.euler_gamma * sigma0
        last_third_data = x[2*len(x)//3:]
        sigma1 = np.std(last_third_data) * np.sqrt(6) / np.pi - sigma0
        mu1 = np.mean(last_third_data) - np.mean(first_third_data) - sigma1 * np.euler_gamma 
    elif model_id == 'mu_rate':
        sigma0 = np.std(x) * np.sqrt(6) / np.pi
        first_third_data = x[:len(x)//3]
        mu0 = np.mean(first_third_data) - np.euler_gamma * sigma0
        last_third_data = x[2*len(x)//3:]
        mu1 = np.mean(last_third_data) - np.mean(first_third_data)
        sigma1 = 0.0
    xi = -0.1
    return xi, mu0, mu1, sigma0, sigma1

def one_over_k_return_level(xi, mu, sigma, k):
    return mu + sigma * ( ( -np.log(1 - 1/k) ) ** (-xi) - 1 ) / xi

def grad_return_level(times, xi, mu0, mu1, sigma, k):
    grad = np.zeros( (4, len(times) ))
    s = -np.log(1 - 1/k)
    s_pow_negxi = s ** (-xi)
    grad[0,:] = - sigma * ( ( s_pow_negxi - 1 ) / xi + s_pow_negxi * np.log(s) ) / xi
    grad[1,:] = 1
    grad[2,:] = times
    grad[3,:] =  ( s_pow_negxi - 1 ) / xi
    #print(grad[:,-1])
    return grad

def return_level_err(times, xi, mu0, mu1, sigma0, k):
    grad = grad_return_level(times, xi, mu0, mu1, sigma0, k)
    ll = lambda x: time_dep_gev_ll('mu_rate', *x)
    res_bis = approx_hess1([xi, mu0, mu1, sigma0], ll)
    cov = np.linalg.inv(-res_bis)
    return_level_var = np.einsum('it,ij,jt->t', grad, cov, grad)
    err = 2 * np.sqrt(return_level_var / XT['x'].size) 
    return err

def chi2_test( ll, ll_nested, delta_df ):
    p = 1 - stats.chi2.cdf( 2 * (ll - ll_nested), delta_df )
    return p

