import numpy as np
from scipy import stats
from astropy.io import fits
from iminuit import Minuit
from pylab import *
from scipy.interpolate import interp1d
from os import walk
from scipy import signal

def func_find_peaks(y, required_dist, required_prominence):  # height, required_dist):
    """identifies peaks, input : 
        - y = an array that contains the signal - the flux/intensity etc; 
        - required_dist = minimum distance between peaks
        - required_prominence = minimum prominence of peaks
        
        returns peak index, prominences, left and right indexes - for widths and the actual width
        [0] peak index in initial array, [1] prominences, [2] left index for widths, [3] right index for widths ,
        [4] length of the width, [5] peak height 0 reference """

    peaks, dict_peak = signal.find_peaks(x=y, distance=required_dist, prominence=required_prominence, height=np.zeros(len(y)))  # height=height,
    prom = signal.peak_prominences(x=y, peaks=peaks, wlen=20)
    widths = signal.peak_widths(x=y, peaks=peaks, rel_height=1, prominence_data=prom)
    return peaks, prom[0], np.round(widths[2]).astype(int), np.round(widths[3]).astype(int), widths[0], dict_peak['peak_heights']


def get_peak_index_ranges(peak_locs, peak_range_size=np.nan):
    """ 
    
    Returns
    -------

        peak_index_ranges : ndarray
            An array of indexes (start, end) of a range around each peak. 

     """
    
    # First compute desired peak_range_size: the mean separation between peaks
    if np.isnan(peak_range_size):
        peak_range_size = int(np.mean(np.diff(peak_locs)))
    
    peak_index_ranges = []
    for nPeak in peak_locs:
        start = nPeak - peak_range_size/2
        end = nPeak + peak_range_size/2
        peak_index_ranges.append([int(start), int(end)])
    return np.asarray(peak_index_ranges)


def fit_peaks(data_spec, data_spec_err, peak_index_ranges, print=False):
    """ Returns array of the fit values with errors: A, A_err, mu, mu_err, sigma, sigma_err, C, C_err, chi2_val, ndof, converged (bool), index_start, index_end, prob """

    peak_fits = []
    for peak_index_range in peak_index_ranges:

        index_start, index_end = peak_index_range

        x = np.arange(index_start, index_end) # range from the start of the peak index to the end of the peak index (index ~ nPix)
        y = np.array(data_spec[index_start:index_end])
        ey = np.array(data_spec_err[index_start:index_end])

        # From Christian
        def super_gauss(x, A, mu, sigma, P, C, b = 0):
            z = (x - mu)**2 / (2 * sigma**2)
            return A * np.exp(-z**P) + C + b * (x - mu)


        # ChiSquare fit model:
        def model_chi(A, mu, sigma, P, C) :
            y_fit = super_gauss(x, A, mu, sigma, P, C)
            chi2 = np.sum(((y - y_fit) / ey)**2)
            return chi2
        model_chi.errordef = 1

        A_init     = 0.87
        mu_init    = np.mean(x)
        sigma_init = -1.8
        P_init     = 1.3
        C_init     = 0.12

        minuit = Minuit(model_chi, A=A_init, mu=mu_init, sigma=sigma_init, P=P_init, C=C_init)

        # Perform the actual fit (and save the parameters):
        m = minuit.migrad()                                             
        # print(m)
        
        # Extract the fitting parameters and their uncertainties:
        mu_fit = minuit.values['mu']
        sigma_mu_fit = minuit.errors['mu']
        Npoints = len(x)
        ndof = Npoints - len(minuit.values[:])
        Chi2_val = minuit.fval # The chi2 value
        converged = minuit.fmin.is_valid
        Prob = stats.chi2.sf(Chi2_val, ndof)


        # peak_fits.append([*minuit.values, Chi2_val])
        peak_fits.append([
            minuit.values['A'],
            minuit.errors['A'],
            minuit.values['mu'],
            minuit.errors['mu'],
            minuit.values['sigma'],
            minuit.errors['sigma'],
            minuit.values['C'],
            minuit.errors['C'],
            Chi2_val,
            ndof,
            converged,
            index_start, 
            index_end,
            Prob,
            minuit.values['P'],
            minuit.errors['P'],
        ])
        
        if print:
            print(f"  Peak fitted. N = {Npoints:2d}   Chi2 ={Chi2_val:5.1f}   Wave mean = {mu_fit:8.3f}+-{sigma_mu_fit:5.3f}")

    return np.asarray(peak_fits)


def get_true_wavel(data_wavel_given, peak_locs):
    """ Takes in the given wavelengths for each pixel and the list of peaks in pixel space.
        
        Returns
        -------

            List of the true wavelengths

     """

    # For a given wavel_given find the closest wavel_true solution 
    # Start by generating at least enough values of true_wavel
        
    def true_wavel(n):
        # True frequency
        c = 299792458   # m/s
        v_rep, v_offset = 14e9, 6.19e9
        f = v_rep * n + v_offset

        # True wavelength in ångstrøm
        wavel = c/f * 1e10
        return wavel

    # Get peaks in wavel_given
    wavel_given = data_wavel_given[peak_locs]
    
    # set max and min 
    wavel_given_min = wavel_given[0]
    wavel_given_max = wavel_given[-1]

    # gen true wavel
    wavel_true = [[true_wavel(0), 0], [true_wavel(1), 1]]
    n = 1
    while wavel_true[-1][0] > wavel_given_min * (1 - 0.001): # continue until we get well below the lowest given wavel. 
        n += 1
        wavel_true.append([true_wavel(n), n])

    # Cut off well above wavel_given_max
    wavel_true = np.asarray(wavel_true)
    wavel_true = wavel_true[wavel_true[:, 0] < wavel_given_max * (1 + 0.001)]
    wavel_true = wavel_true[::-1] # reverse order

    # wavel_true list is now quite a bit longer than peak_locs, but contains all of them
    # Now, find the closest match in wavel_true for each value in 
    wavel_true_match = []
    for lambd_given in wavel_given:
        # Find minimum value
        # wavel_true_match.append(min(wavel_true[:, 0], key=lambda x:abs(x-lambd_given)))
        
        # Find minimum value
        min_val = min(wavel_true[:, 0], key=lambda x:abs(x-lambd_given))

        # Find index of minimum value
        min_val_index = next((idx for idx, val in np.ndenumerate(wavel_true[:, 0]) if val==min_val), None)[0]

        # So we can find the n of the peak
        n = wavel_true[:, 1][min_val_index]

        # now add to list
        wavel_true_match.append([min_val, n])

    return np.asarray(wavel_true_match)


def get_calib_poly_func(degree):
    assert (degree == np.arange(1, 10)).any(), "Not within 1-9 degrees"
    exec(f"function = calib_poly_func_{degree}", globals())
    return function

def calib_poly_func_1(x, c0, c1):
    return c0 + c1*x

def calib_poly_func_2(x, c0, c1, c2):
    return c0 + c1*x + c2*x**2

def calib_poly_func_3(x, c0, c1, c2, c3):
    return c0 + c1*x + c2*x**2 + c3*x**3

def calib_poly_func_4(x, c0, c1, c2, c3, c4):
    return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4

def calib_poly_func_5(x, c0, c1, c2, c3, c4, c5):
    return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5

def calib_poly_func_6(x, c0, c1, c2, c3, c4, c5, c6):
    return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6

def calib_poly_func_7(x, c0, c1, c2, c3, c4, c5, c6, c7):
    return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6 + c7*x**7

def calib_poly_func_8(x, c0, c1, c2, c3, c4, c5, c6, c7, c8):
    return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6 + c7*x**7 + c8*x**8

def calib_poly_func_9(x, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9):
    return c0 + c1*x + c2*x**2 + c3*x**3 + c4*x**4 + c5*x**5 + c6*x**6 + c7*x**7 + c8*x**8 + c9*x**9


def fit_peak_positions(wavel_true_match, peak_fits, poly_degree):
    """ Fits LFC peak locations with a polynomial of specified degree (up to 9th) """

    x = peak_fits[:,2]
    y = wavel_true_match
    y_err = peak_fits[:, 4]    # sigma (width of the peak)

    fit_func = get_calib_poly_func(poly_degree)
    init_values = np.ones(poly_degree + 1)

    # Quad fit
    model_chi2 = Chi2Regression(fit_func, x, y, y_err)
    model_chi2.errordef = 1

    # Fit peak with a Gaussian:
    minuit = Minuit(model_chi2, *init_values)

    # Perform the actual fit (and save the parameters):
    minuit.migrad()                                             
        
    # Extract the fitting parameters and their uncertainties:
    Npoints = len(x)
    Nvar = poly_degree + 1                               # Number of variables
    Ndof_fit = Npoints - Nvar                       # Number of degrees of freedom = Number of data points - Number of variables
    chi2_fit = minuit.fval                          # The chi2 value
    prob_fit = stats.chi2.sf(chi2_fit, Ndof_fit)    # The chi2 probability given N degrees of freedom
    # print(f"  Peak fitted. N = {Npoints:2d}   Chi2 ={Chi2_fit:5.1f}")
    return minuit.values, chi2_fit, prob_fit, fit_func, minuit.valid


def fit_all_peaks_in_all_orders(filename = r"expres_tp/LFC_200907.1063.fits", correct_errors=False, custom_error_factor=None):
    """ Load LFC exposure, find peaks and fits with super-Gaussian to find precice peak location. """

    # Load data
    hdu1 = fits.open(filename)

    data = hdu1[1].data.copy()

    results = []
    # for order in range(0, len(data)):
    for order in range(40, 76):
        data_spec       = data['spectrum'][order]
        data_spec_err   = data['uncertainty'][order]
        data_wavel      = data['wavelength'][order]

        if correct_errors:
            if custom_error_factor is not None:
                data_spec_err = data_spec_err * custom_error_factor
            else:
                data_spec_err = data_spec_err * np.sqrt(3)

        # Find peaks
        peak_info = func_find_peaks(data_spec, 11, 0.15)
        peak_locs = peak_info[0]

        # If less than 10 peaks skip order
        if len(peak_locs) < 10:
            results.append([order, [[np.nan]], [[np.nan]], [np.nan] ]) # save NaN to list if no peak
            continue

        # Create data slices around each peak
        peak_index_ranges = get_peak_index_ranges(peak_locs)

        # Fit peak in each data slice
        peak_fits = fit_peaks(data_spec, data_spec_err, peak_index_ranges)

        # Get list of true wavelengths
        wavel_true = get_true_wavel(data_wavel, peak_locs)
        
        results.append([order, peak_fits, wavel_true, data_wavel[peak_locs]])

    results = np.asarray(results, dtype=object)
    return results


# ========================= iminuit things ==============================

from iminuit.util import make_func_code
from iminuit import describe #, Minuit,

def set_var_if_None(var, x):
    if var is not None:
        return np.array(var)
    else: 
        return np.ones_like(x)
    
def compute_f(f, x, *par):
    
    try:
        return f(x, *par)
    except ValueError:
        return np.array([f(xi, *par) for xi in x])


class Chi2Regression:  # override the class with a better one
        
    def __init__(self, f, x, y, sy=None, weights=None, bound=None):
        
        if bound is not None:
            x = np.array(x)
            y = np.array(y)
            sy = np.array(sy)
            mask = (x >= bound[0]) & (x <= bound[1])
            x  = x[mask]
            y  = y[mask]
            sy = sy[mask]

        self.f = f  # model predicts y for given x
        self.x = np.array(x)
        self.y = np.array(y)
        
        self.sy = set_var_if_None(sy, self.x)
        self.weights = set_var_if_None(weights, self.x)
        self.func_code = make_func_code(describe(self.f)[1:])

    def __call__(self, *par):  # par are a variable number of model parameters
        
        # compute the function value
        f = compute_f(self.f, self.x, *par)
        
        # compute the chi2-value
        chi2 = np.sum(self.weights*(self.y - f)**2/self.sy**2)
        
        return chi2


