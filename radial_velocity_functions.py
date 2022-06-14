import numpy as np
from scipy import stats
from astropy.io import fits
from iminuit import Minuit
from pylab import *
from scipy.interpolate import interp1d
from os import walk
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from calibration_functions import *
from tqdm import tqdm
import pandas as pd

my_colors_red = "C3"
my_colors_blue = "C0"

def make_nan_matrix(size):
    matrix = np.empty((size,size))
    matrix[:] = np.nan
    return matrix

def weighted_mean(x, errors):
    m1 = np.sum([x/s**2 for x, s in zip(x, errors)])
    m2 = np.sum([1/(x**2) for x in errors])
    mean = m1/m2
    err = np.sqrt(1/np.sum([1/(x**2) for x in errors]))
    return (mean, err)

def compute_rms(x):
    x = x - np.mean(x)
    return np.sqrt(np.sum(x ** 2) / len(x - 1))


def compute_coords(filenames):
    size = len(filenames)
    coords = []
    for x in np.arange(size):
        for y in np.arange(x, size):
            if x != y:
                coords.append((x, y)) 
    return coords


SPECTRA_PATH_34411 = "/Users/jakobharteg/Data/34411_spectra/"
SPECTRA_PATH_10700 = "/Users/jakobharteg/Data/10700_spectra/"
SPECTRA_PATH_26965 = "/Users/jakobharteg/Data/26965_spectra/"
SPECTRA_PATH_101501 = "/Users/jakobharteg/Data/101501_spectra/"
SPECTRA_PATH_51PEG = "/Users/jakobharteg/Data/51peg/"

def get_all_spectra_filenames(spectra_path = SPECTRA_PATH_34411):
    """ Returns all filenames of the spectra files in the path speficied in spectra_path"""
    SPEKTRA_filenames = next(walk(spectra_path), (None, None, []))[2]  # [] if no file
    SPEKTRA_filenames = sorted(SPEKTRA_filenames)
    SPEKTRA_filenames = [spectra_path + x for x in SPEKTRA_filenames] # add path to the filename
    return SPEKTRA_filenames

def load_spectra_fits(filename):
    """ Returns data from a fits file with a given name """
    hdul = fits.open(filename)
    data = hdul[1].data.copy()
    hdul.close()
    return data

def get_spec_wavel(data, order, continuum_normalized=False, bary_corrected=True):
    """ Returns intensity, intensity_err and wavelength for a given spectra data.
        Use in conjunction with load_spectra_fits """

    excalibur_mask  = data['EXCALIBUR_MASK'][order]    # filter by EXCALIBUR_MASK
    data_spec       = data['spectrum'][order][excalibur_mask]
    data_spec_err   = data['uncertainty'][order][excalibur_mask]
    
    if bary_corrected:
        data_wavel = data['BARY_EXCALIBUR'][order][excalibur_mask]
    else:
        data_wavel = data['EXCALIBUR'][order][excalibur_mask]
    
    if continuum_normalized:
        cont = data['continuum'][order][excalibur_mask]
        data_spec = data_spec / cont
        data_spec_err = data_spec_err / cont

    return data_spec, data_spec_err, data_wavel


def get_spectra_seconds_since_epoch(filename):
    from datetime import datetime
    dt = get_spektra_date_and_time(filename)
    dt_obj = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    return dt_obj.timestamp()


def get_spektra_date_and_time(filename):
    """ Returns the date and time of observation for a given fits filename """
    hdul = fits.open(filename)
    header = hdul[0].header
    hdul.close()
    date = header["DATE-OBS"]
    return date


def get_spektra_date(filename):
    """ Returns the date of observation for a given fits filename """
    date = get_spektra_date_and_time(filename)
    date = date[:date.index(" ")]
    return date


def get_spectra_dates(filenames):
    """ Returns a list of dates (year, month, date) for given list of fits filenames """
    dates = []
    for i in np.arange(len(filenames)):
        date = get_spektra_date(filenames[i])
        year, month, date = date.split("-")
        year, month, date = int(year), int(month), int(date)
        dates.append((year, month, date))
    return dates


def convert_dates_to_relative_days(dates):
    from datetime import datetime
    """ Returns a list of days relative to the first day """
    days = []
    first_day = datetime(*dates[0])
    for i in np.arange(len(dates)):
        d1 = datetime(*dates[i])
        dd1 = d1 - first_day
        days.append(dd1.days)
    return np.asarray(days)


# There are often several observations per night
def get_spectra_filenames_without_duplicate_dates(spectra_path = SPECTRA_PATH_34411):
    """ Returns a list of filesnames of the spectra files in the path speficied in SPECTRA_PATH_34411 without date-duplicates, i.e.
        oberservations taken on the same day. """
    all_files = get_all_spectra_filenames(spectra_path)
    all_dates = get_spectra_dates(all_files)
    files = [all_files[0]]
    dates = [all_dates[0]]
    for i in np.arange(1, len(all_dates)):
        if dates[-1] != all_dates[i]:
            dates.append(all_dates[i])
            files.append(all_files[i])
    return files


def plot_spectra_dates(spectra_dates, label=None):
    from datetime import datetime
    plt.figure(figsize=(25, 8))
    for index, date in enumerate(spectra_dates):
        year, month, date = date
        d = datetime(year, month, date)
        if label and index == len(spectra_dates) - 1:
            plt.scatter(d, index, color="k", s=2, label=label)
            plt.legend()
        else:
            plt.scatter(d, index, color="k", s=2)
        

def plot_spectra_dates_from_path(path):
    filenames = get_all_spectra_filenames(path)
    dates = get_spectra_dates(filenames)
    plot_spectra_dates(dates, label=f"{len(dates)} files found in {path}")



def plot_matrix(diff_matrix, diff_matrix_err=None, diff_matrix_valid=None, plot_ratio=True, kms=False, colormap1="coolwarm", colormap2="Blues", save_as=None):
    """ Plot shift matrices """

    if plot_ratio:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(8,3))
    else:
        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8,4))

    def fix_grid_lines(ax, size):
        pass
        # ax.hlines(y=np.arange(0, size)+0.5, xmin=np.full(size, 0)-0.5, xmax=np.full(size, size)-0.5, color="black", alpha=0.2)
        # ax.vlines(x=np.arange(0, size)+0.5, ymin=np.full(size, 0)-0.5, ymax=np.full(size, size)-0.5, color="black", alpha=0.2)

    if kms:
        diff_matrix = diff_matrix[:] / 1000

    # ======= PLOT 1 ============ Mean shifts
    cs = ax1.imshow(diff_matrix, cmap=plt.get_cmap(colormap1))
    cax = make_axes_locatable(ax1).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
    cbar = fig.colorbar(cs, ax=ax1, cax=cax)
    cbar.set_label('m/s', rotation=270, labelpad=15)
    if kms:
        cbar.set_label('km/s', rotation=270)
    fix_grid_lines(ax1, len(diff_matrix))
    ax1.set_title("$\Delta V_r^{ij}$ : Relative RV")
    ax1.set_xlabel("$i$")
    ax1.set_ylabel("$j$", rotation=0, labelpad=10)

    # ======= PLOT 2 ============ Errors
    if diff_matrix_err is not None:

        cs = ax2.imshow(diff_matrix_err, cmap=plt.get_cmap(colormap2))
        cax = make_axes_locatable(ax2).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
        cbar = fig.colorbar(cs, ax=ax2, cax=cax)
        cbar.set_label('m/s', rotation=270, labelpad=15)
        fix_grid_lines(ax2, len(diff_matrix_err))
        ax2.set_title("$\sigma (\Delta V_r^{ij})$ : RV errors")
        ax2.set_xlabel("$i$")
        ax2.set_ylabel("$j$", rotation=0, labelpad=10)

    # ======= PLOT 3 ============ Valid/Convergence ratio
    if diff_matrix_valid is not None:
        if plot_ratio:
            cs = ax3.imshow(diff_matrix_valid)
            cax = make_axes_locatable(ax3).append_axes('right', size='5%', pad=0.05) # to make color bar same height at plot, needed when making several subplots with each colorbar
            cbar = fig.colorbar(cs, ax=ax3, cax=cax)
            fix_grid_lines(ax3, len(diff_matrix_valid))
            ax3.set_title("Valid ratio")
            # Fix spacing between plots
            fig.subplots_adjust(wspace=0.25)


    fig.subplots_adjust(wspace=0)
    fig.tight_layout()

    if save_as is not None:
        fig.savefig(save_as, bbox_inches="tight", dpi=300)

def find_all_features(filenames, custom_error_scale, use_bary_correction):
    features = []
    for filename in tqdm(filenames):
        features.append(find_features(filename, log=False, custom_error_scale=custom_error_scale, use_bary_correction=use_bary_correction))
    return features


def find_features(filename, 
                    plot_orders = None, 
                    plot_features_in_order = None, 
                    log=True, 
                    max_frac_err = 0.1,                 # maximum fractional error in intensity
                    min_order_goodness = 0.7,           # Min fraction of data in an order that should be left after filtering for the order to be included. 
                    # min_peak_dist = 50,               # minimum distance (in pixels) between peaks  
                    min_peak_dist = 5,                  # minimum distance (in pixels) between peaks  
                    min_peak_prominence = 0.25,         # minimum height of peak from base (not zero)
                    is_51_peg = False,                  # 51Peg doesn't have excalibur column so we need different keywords 
                    use_bary_correction = True,         # True if you want to use "wavelength" column filtered by pixel_mask.
                    scale_up_errors = False,            # scale errors by sqrt{3}
                    custom_error_scale = None           # scale errors by a constant
    ):
    
    """ Returns list of features x_values (angstrom by default), y_values, y_err_values, x_peak_location, peak_index, order """
    """                          0                               1         2             3                4           5     """

    
    feature_slices = []
    fits_data = load_spectra_fits(filename)
    orders_n = shape(fits_data["spectrum"])[0]

    # 51Peg data files have different column names
    is_51_peg = ("peg" in filename)

    orders = np.arange(0, orders_n)
    for order in orders:
        
        if is_51_peg:
            pixel_mask  = fits_data['PIXEL_MASK'][order]
            y           = fits_data['spectrum'][order][pixel_mask]
            og_y        = y # copy of original y data before filtering
            y_err       = fits_data['uncertainty'][order][pixel_mask]
            continuum   = fits_data['continuum'][order][pixel_mask]
            x           = fits_data['bary_wavelength'][order][pixel_mask]
        elif not use_bary_correction:
            # pixel_mask  = fits_data['PIXEL_MASK'][order]    # filter by pixel_mask
            # y           = fits_data['spectrum'][order][pixel_mask]
            # og_y        = y # copy of original y data before filtering
            # y_err       = fits_data['uncertainty'][order][pixel_mask]
            # continuum   = fits_data['continuum'][order][pixel_mask]
            # x           = fits_data['wavelength'][order][pixel_mask]

            excalibur_mask  = fits_data['EXCALIBUR_MASK'][order]    # filter by EXCALIBUR_MASK
            y           = fits_data['spectrum'][order][excalibur_mask]
            og_y        = y # copy of original y data before filtering
            y_err       = fits_data['uncertainty'][order][excalibur_mask]
            continuum   = fits_data['continuum'][order][excalibur_mask]
            x           = fits_data['excalibur'][order][excalibur_mask]
        else:
            excalibur_mask  = fits_data['EXCALIBUR_MASK'][order]    # filter by EXCALIBUR_MASK
            y           = fits_data['spectrum'][order][excalibur_mask]
            og_y        = y # copy of original y data before filtering
            y_err       = fits_data['uncertainty'][order][excalibur_mask]
            continuum   = fits_data['continuum'][order][excalibur_mask]
            x           = fits_data['BARY_EXCALIBUR'][order][excalibur_mask]

        # skip order if no good data
        if len(x) == 0 or len(y) == 0:
            continue

        if scale_up_errors:
            y_err = y_err * np.sqrt(3)

        if custom_error_scale is not None:
            y_err = y_err * custom_error_scale

        # Normalize intensity by continuum 
        y = y/continuum
        y_err = y_err/continuum

        # filter by fractional error (10% doesn't actually remove anything)
        frac_err = y_err/y
        frac_err_mask = (0 < frac_err) & (frac_err < max_frac_err) # reject if larger than 10% and negative
        y = y[frac_err_mask]
        y_err = y_err[frac_err_mask]
        x = x[frac_err_mask]

        # Skip order if we filtered out more than 1 - min_order_goodness of the data (bad order ... )
        if len(y) / len(og_y) < min_order_goodness:
            # print("Order bad:", len(y), "/", len(og_y), "=", len(y) / len(og_y))
            continue

        # Now invert peaks
        y = 1 - y

        peaks = func_find_peaks(y, min_peak_dist, min_peak_prominence)
        peak_locs = peaks[0]
        peak_height = peaks[5] # peak height from y=0 
        
        # Plot
        if plot_orders is not None and (plot_orders == order).any() or plot_orders == -1:
            plt.figure(figsize=(16,3))
            plt.plot(x, y, ".")
            plt.plot(x[peak_locs], peak_height, "o", color="C3", label=f"{order}. order")
            plt.ylabel("1 - Continuum normalized counts")
            plt.xlabel("Wavelength [Å]")
            plt.legend(loc = "upper right")
        
        # If less than 10 peaks skip order
        if len(peak_locs) < 5:
            continue

        peak_index_ranges = get_peak_index_ranges(peak_locs, peak_range_size=30)

        # feature_slices = []
        for index, range in enumerate(peak_index_ranges):
            start, end = range

            if len(x[start:end]) != 0 and len(y[start:end]) != 0: # sometimes these lists come out empty. 
                feature_slices.append([x[start:end], y[start:end], y_err[start:end], x[peak_locs[index]], peak_locs[index], order])

            if plot_features_in_order is not None and (plot_features_in_order == order).any():
                plt.figure(figsize=(10,3))
                plt.plot(x[start:end], y[start:end], ".")
                plt.plot(x[peak_locs[index]], peak_height[index], "o", color="C3", label=f"$k = {index}$")
                plt.legend(loc = "upper right")
        
    if log:
        print(len(feature_slices), "peaks found")

    return np.asarray(feature_slices, dtype=object)


def find_feature_matches(features1, features2, log=True, filter=True, relative_tolerance=0.0003):
    """ Finds features in two files, and iterates from the lowest wavelength to the highest, in small
        steps, asking if either of the files have features close to the current wavelength within a
        defined tolerance. If so, it returns a match. 
    """

    peaks1 = np.array(features1[:, 3], dtype=float)
    peaks2 = np.array(features2[:, 3], dtype=float)

    max1, max2 = np.nanmax(peaks1), np.nanmax(peaks2)
    min1, min2 = np.nanmin(peaks1), np.nanmin(peaks2)

    common_max = min([max1, max2])
    common_min = max([min1, min2])

    # We should of course only use each peak once, so let's keep track of the peaks we use
    used_peaks1 = np.asarray([False] * len(peaks1))
    used_peaks2 = np.asarray([False] * len(peaks2))

    range = np.linspace(common_min, common_max, 10000)
    # relative_tolerance = 0.0001 # lower value --> fewer matches, higher --> worse matches # but first we set a high value and then filter later
    # relative_tolerance = 0.00005 # lower value --> fewer matches, higher --> worse matches
    # relative_tolerance = 0.001 # lower value --> fewer matches, higher --> worse matches
    # relative_tolerance = 0.00002 # lower value --> fewer matches, higher --> worse matches
    
    matches = []
    for wavel in range:
        
        # see if we have a match in the two lists:
        match_mask1 = np.isclose(peaks1, wavel, relative_tolerance)
        match_mask2 = np.isclose(peaks2, wavel, relative_tolerance)
        
        # Only take match if we have only one match from each file
        if len(match_mask1[match_mask1]) == 1 and len(match_mask2[match_mask2]) == 1:

            # get the two features that matched
            f1 = features1[match_mask1][0]
            f2 = features2[match_mask2][0]

            # Now perform a bit of filtering: 
            if filter:

                # Check if the sum of the y values are about equal, if so it means we probably have the same peak.
                f1_spec, f2_spec = f1[1], f2[1]
                if np.isclose(np.sum(f1_spec), np.sum(f2_spec), 0.1) == False:
                    continue

                # If the difference bewteen the two peaks is larger than 5 cm/s pass it
                # f1_peak, f2_peak = f1[3], f2[3]
                # if abs(f1_peak - f2_peak) > 5:
                #     continue
                    
            # Get the index of the peak
            used_peak_index2 = np.where(match_mask2)[0]
            used_peak_index1 = np.where(match_mask1)[0]

            # Append if we have not already used this peak
            if used_peaks1[used_peak_index1] == False and used_peaks2[used_peak_index2] == False:
                
                # Append
                matches.append([f1, f2])

                # Mark that we used the peaks
                used_peaks1[used_peak_index1] = True
                used_peaks2[used_peak_index2] = True
            
    if log:
        print(f"{len(matches)} matches found")
    
    return matches


def nth_cloest_match(value, array, n):
    """ Returns the index of the nth closest match for a given value and a given array"""
    diff_array = np.abs(array - value)
    return np.argpartition(diff_array, n)[n]


def find_feature_matches2(features1, features2, log=True, filter=True, max_dist=0.5, max_area_diff=0.2):
    """ This feature match finder simply returns the closest match between two lists of features.
        """

    peaks1 = np.array(features1[:, 3], dtype=float)
    peaks2 = np.array(features2[:, 3], dtype=float)

    # We should of course only use each peak once, so let's keep track of the peaks we use
    used_peaks1 = np.asarray([False] * len(peaks1))
    used_peaks2 = np.asarray([False] * len(peaks2))

    # find matches
    matches = []
    N_reject_dist = 0
    N_reject_area = 0
    # TODO: repeat until all peaks are used
    for i in np.arange(len(peaks1)):
        peak1 = peaks1[i]
        min_index = nth_cloest_match(peak1, peaks2, 0)

        # define feature
        f1, f2 = features1[i], features2[min_index]

        if filter:
            # Check if peak wavel location diff is larger than 0.5 A, that would be equivalant to about 30 km/s
            # (very generous cut, because, since the data is not continous sometimes the difference is quite large, 
            # before we do the cross correlation)
            if max_dist != -1:
                if np.abs(f1[3] - f2[3]) > max_dist:
                    N_reject_dist += 1
                    continue

            # Check if the integral/area under the graph of the peaks is about the same
            if max_area_diff != -1:
                if np.abs(np.sum(f1[1]) - np.sum(f2[1])) > max_area_diff:
                    N_reject_area += 1
                    continue
        
        # Append if we have not already used this peak
        if used_peaks1[i] == False and used_peaks2[min_index] == False:
            
            # Append
            matches.append([f1, f2])

            # Mark that we used the peaks
            used_peaks1[i] = True
            used_peaks2[min_index] = True
    matches = np.asarray(matches)

    if log:
        print(f"{len(matches)} matches found : dist rejected {N_reject_dist}, area rejected {N_reject_area}")
        # print(f"Rejected {N_reject_dist} proposed matches")
        # print(f"Rejected {N_reject_area} proposed matches based on area")

    return matches


def plot_feature_matches2(matches, mask):
    """ Plots an overview of the peak location difference from features matches with a valid/invalid mask """
    plt.figure(figsize=(10, 16))
    peak_loc_diffs = np.asarray([f1[3] - f2[3] for f1, f2 in matches])
    y = np.arange(len(peak_loc_diffs))
    plt.plot(peak_loc_diffs[mask], y[mask], ".", color="C2", label=f"{len(peak_loc_diffs[mask])} valid matches")
    plt.plot(peak_loc_diffs[mask == False], y[mask ==  False], ".", color="Red", label=f"{len(peak_loc_diffs[mask==False])} invalid matches")
    plt.xlabel("Peak shift [A]")
    plt.ylabel("Peak index")
    plt.gca().invert_yaxis()
    plt.legend(bbox_to_anchor=(1.45, 0.99))
    plt.title("Initial peak match selection")


def compute_match_filter_mask(matches, max_dist, max_area_diff):
    """ Returns a mask with TRUE if matches pass the filter of max_dist and max_area_diff """
    mask = []
    for match in matches:
        f1, f2 = match[0], match[1]
        dist_ok = np.abs(f1[3] - f2[3]) < max_dist
        area_ok = np.abs(np.sum(f1[1]) - np.sum(f2[1])) < max_area_diff
        mask.append((dist_ok and area_ok))
    return np.asarray(mask)


def compute_feature_shift(x1, y1, y1_err, peak1, x2, y2, peak2, plot=False, ax=None, return_df=False, interp_size = 1000, return_extra=False):
    """ Attempts to fit two features with based on a shift parameter.
        Returns shift_min_final (m/s), shift_min_final_err (m/s), valid 
        """
    
    c = 299792458 # m/s
    G = 6.67e-11 # N m^2 / kg^2
    M_sun = 2e30
    R_sun = 696340000
    M_star = 1.08 * M_sun # kg
    R_star = 1.28 * R_sun
    G_pot = G*M_star/R_star

    # Interp first file
    f1 = interp1d(x1, y1, kind='cubic', fill_value="extrapolate")
    f1_err = interp1d(x1, y1_err, kind='cubic', fill_value="extrapolate")

    # ChiSquare fit model:
    def model_chi2(A):

        # Interpolate template
        # interp_x2 = x2 * (1 + A/c)/(1 - G_pot/c**2 ) # with GR ... neglibible
        interp_x2 = x2 * (1 + A/c) # Wavelengths are be stretched by a factor of (1 + v/c)
        f2 = interp1d(interp_x2, y2, kind='cubic', fill_value="extrapolate")

        # Find common x-range
        xmin = max([min(x1), min(interp_x2)])
        xmax = min([max(x1), max(interp_x2)])
        xnewCommon = np.linspace(xmin, xmax, interp_size)
        
        # Evaluate interpolation
        ynew1 = f1(xnewCommon)
        ynew2 = f2(xnewCommon)
        
        # Interpolate errors
        ynew1_err = f1_err(xnewCommon)

        # Compute chi2
        chi2 = np.sum(((ynew1 - ynew2) / ynew1_err)**2)
        return chi2
    model_chi2.errordef = 1
        
    # Init value
    # A_init = (peak2 / peak1 - 1 ) * c # shift between the two peaks
    A_init = (peak1 / peak2 - 1 ) * c # shift between the two peaks

    # Compute bounds on A
    x1_min, x1_max = min(x1), max(x1)
    A_lower_bound = (x1_min / peak2 - 1 ) * c
    A_upper_bound = (x1_max / peak2 - 1 ) * c

    minuit = Minuit(model_chi2, A=A_init)
    minuit.limits["A"] = (A_lower_bound, A_upper_bound)
    minuit.migrad()
    
    # Results
    valid = minuit.valid
    shift_min_final = minuit.values['A']
    shift_min_final_err = minuit.errors['A']
    forced = minuit.fmin.has_made_posdef_covar
    at_limit = minuit.fmin.has_parameters_at_limit

    if forced or at_limit:
        valid = False

    # Plot final shifted values
    if plot:

        if ax == None:
            fig, ax = plt.subplots(figsize=(14,6))

        ax.plot(x1, y1)
        ax.plot(x2 * (1 + shift_min_final/c), y2)

        if not valid:
            ax.set_facecolor('pink')


    if return_df:
        return pd.DataFrame({
            'shift':    [shift_min_final],
            'err':    [shift_min_final_err],
            'valid': [valid],
            })
    elif return_extra:

        # Return peak locations of the two features : takes a lot of extra time
        # # Interpolate and return the peak wavelength (should be better than nearest pixel)
        # f1 = interp1d(x1, y1, kind='cubic', fill_value="extrapolate")
        # f2 = interp1d(x2, y2, kind='cubic', fill_value="extrapolate")
        # new_x1 = np.linspace(min(x1), max(x1), 1000)
        # new_x2 = np.linspace(min(x2), max(x2), 1000)
        # interp_y1 = f1(new_x1)
        # interp_y2 = f2(new_x2)
        # new_peak1_wavel = new_x1[np.argmax(interp_y1)]
        # new_peak2_wavel = new_x2[np.argmax(interp_y2)]

        # return shift_min_final, shift_min_final_err, valid, minuit.fval, new_peak1_wavel, new_peak2_wavel
        
        return shift_min_final, shift_min_final_err, valid, minuit.fval
    else:
        return shift_min_final, shift_min_final_err, valid#, minuit


def compute_all_feature_shifts(matches, log=True, plot=False, ax=[], interp_size = 1000, return_extra=False, return_df=False):
    """ Calls compute_feature_shift for a list of matches """

    shifts = []
    # for k in tqdm(np.arange(len(matches))):
    for k in np.arange(len(matches)):
        f1 = matches[k][0]
        f2 = matches[k][1]

        x1      = f1[0]
        y1      = f1[1]
        y1_err  = f1[2]
        peak1   = f1[3]
        x2      = f2[0]
        y2      = f2[1]
        peak2   = f2[3]

        if len(ax) == 0:
            shifts.append(compute_feature_shift(x1, y1, y1_err, peak1, x2, y2, peak2, plot=plot, interp_size = interp_size, return_df=return_df, return_extra=return_extra))
        else:
            shifts.append(compute_feature_shift(x1, y1, y1_err, peak1, x2, y2, peak2, plot=plot, ax=ax[k], interp_size = interp_size, return_df=return_df, return_extra=return_extra))


    shifts = np.asarray(shifts, dtype=object)
    if log:
        failed_n = len(shifts[:, 2][shifts[:, 2] == 0])
        if failed_n > 0: # only print if some fits failed
            print(f"{failed_n} / {len(shifts)} fits failed")
    return shifts


def analyse_and_plot_shifts(path, file_index1, file_index2, bary=True):
    """ Short hand function to compute and plot shifts between two files """

    filenames = get_spectra_filenames_without_duplicate_dates(path)
    file1, file2 = filenames[file_index1], filenames[file_index2]
    matches = find_feature_matches2(find_features(file1, use_bary_correction=bary), find_features(file2, use_bary_correction=bary))

    # Compute how many rows we need to display all
    height = 1
    while 10 * height < len(matches):
        height += 1
    assert (height < 100), "Height is more than 100"

    # Compute a plot shifts
    fig, axs = plt.subplots(height, 10, figsize=(20, 20))
    shifts = compute_all_feature_shifts(matches, plot=True, ax=axs.flat)

    # Remove ticks
    for ax in axs.flat:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    # Plot shift values
    plot_features_shift(shifts)

    return shifts


def plot_features_shift(shifts, ax=None, labels=True, title=None, legend=True, side_info = True, draw_guides=True, guide_lw=0.5, side_text_y_loc=0.84, legend_x_loc=1.4):

    s, s_err, s_valid = shifts[:, 0], shifts[:, 1], shifts[:, 2]

    feature_n = np.arange(len(s))
    valid_shifts, valid_shifts_err, valid_features          = s[s_valid == 1], s_err[s_valid == 1], feature_n[s_valid == 1]
    invalid_shifts, invalid_shifts_err, invalid_features    = s[s_valid == 0], s_err[s_valid == 0], feature_n[s_valid == 0]

    if ax == None:
        fig, ax = plt.subplots(figsize=(8, 12))

    ax.errorbar(valid_shifts, valid_features, xerr=valid_shifts_err, fmt=".", ms=1, linewidth=1, color=my_colors_blue, label=rf'{len(valid_shifts)} feature shifts with error')    
    # ax.errorbar(invalid_shifts, invalid_features, xerr=invalid_shifts_err, fmt="none", linewidth=1, color = "C3", label=rf'{len(invalid_shifts)} invalid fits, with errors')

    # ax.scatter(valid_shifts, valid_features, s=1, color=my_colors_blue)
    # ax.scatter(invalid_shifts, invalid_features, s=1, color=my_colors_red)

    shift_mean, shift_mean_err = weighted_mean(valid_shifts, valid_shifts_err)
    shift_mean_np = np.mean(valid_shifts)
    shift_mean_np_err = np.std(valid_shifts) / np.sqrt(len(valid_shifts))
    median = np.median(valid_shifts)
    median_err = shift_mean_np_err * np.sqrt(np.pi/2)

    # # Plot mean and err
    if draw_guides:
        ax.vlines(shift_mean, -20, len(s) + 20, linestyle="-", alpha=1.0, color="orange", label="Weighted average", lw=guide_lw, zorder=100)
        ax.vlines(shift_mean_np, -20, len(s) + 20, linestyle="-", alpha=1.0, color="black", label="Mean", lw=guide_lw, zorder=100)
        ax.vlines(median, -20, len(s) + 20, linestyle="-", alpha=1.0, color=my_colors_red, label="Median", lw=guide_lw, zorder=100)
    # ax.axvspan(shift_mean-shift_mean_err, shift_mean+shift_mean_err, alpha=0.2) # too small to see anyway

    # invert axis so feature 0 starts at the top
    plt.gca().invert_yaxis()

    if labels:
            if legend:
                ax.legend(bbox_to_anchor=(legend_x_loc, 0.99))

            ax.set_xlabel("Velocity shift [m/s]")
            ax.set_ylabel("Feature")
            
            if side_info:
                std_err = np.std(valid_shifts)/np.sqrt(len(valid_shifts))
                text = f'''Weighted average = ({shift_mean:.2f} ± {shift_mean_err:.1}) m/s 
                        \n Mean = ({shift_mean_np:.2f} ± {shift_mean_np_err:.1f} ) m/s
                        \n Median = ({median:.2f} ± {median_err:.1f}) m/s'''
                        # \n std = {np.std(valid_shifts):.1f} m/s
                        # \n std/sqrt(N) = {std_err:.1f} m/s
                        
                ax.text(1.08, side_text_y_loc, text,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax.transAxes)
            
            if title is not None:
                ax.set_title(title)
                
            # ax.set_title("all_features_34411_ms_non_bary.npy for coord 42")
    else:
            # hide ticks
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])


    # return fig
    # fig.savefig("fdsfds.png", dpi=400)



def plot_features_shift_matrix(result, coords, save_to_filename=None, do_not_show=False):
    size = np.max(np.max(coords)) + 1
    fig, axs = plt.subplots(size, size, figsize=(size, size))

    # Hide labels on all 
    for ax in axs.flat:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    for x in np.arange(size):
        for y in np.arange(x, size):
            
            # Get index from coords
            try:
                index = list(coords).index((x, y))
                plot_features_shift(result[index], axs[x, y], labels=False)
            except:
                continue


    # Remove space between graphs
    plt.subplots_adjust(wspace=0, hspace=0)

    # Write column and row numbers
    cols = [col for col in range(0, size)]
    rows = [row for row in range(0, size)]
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)
    for ax, row in zip(axs[:,0], rows):
        ax.set_ylabel(f"{row}     ", rotation=0, size='large')

    if save_to_filename != None:
        plt.savefig(f"figures/{save_to_filename}", dpi=100)

    if do_not_show:
        plt.close(fig) # don't show figure


def parse_matrix_results(result, coords, median_err=False, use_median=True):
    
    size = np.max(np.max(coords)) + 1
    diff_matrix, diff_matrix_err, diff_matrix_valid = make_nan_matrix(size), make_nan_matrix(size), make_nan_matrix(size)

    for coord, shifts in zip(coords, result):

        # Split 
        rv, err, valid = shifts[:, 0], shifts[:, 1], shifts[:, 2]

        # Compute valid ratio, number of succesfull fits / total number of fits
        valid_ratio = len(valid[valid == 1])/len(valid)

        rv_valid = rv[valid == 1]
        rv_valid_err = err[valid == 1]
        
        # Compute weighted average
        rv, err = weighted_mean(rv_valid, rv_valid_err)

        if use_median:
            rv = np.median(rv_valid)

        if median_err:
            err = np.std(rv_valid) / np.sqrt(len(rv_valid)) * np.sqrt(np.pi/2)

        x = coord[0]
        y = coord[1]

        diff_matrix[x, y] = rv
        diff_matrix_err[x, y] = err
        diff_matrix_valid[x, y] = valid_ratio
        
    return diff_matrix, diff_matrix_err, diff_matrix_valid


def get_above_diagonal(matrix):
    """ returns returns the above diagonal in a mtrix """
    size = matrix.shape[0]
    off_diag = []
    for x in np.arange(0, size):
        if x + 1 < size:
            off_diag.append(matrix[x, x + 1])
    off_diag = np.asarray(off_diag)
    return off_diag


def compute_z_test_mask(x, sigma):
    """ Returns a mask with TRUEs if values are within the given number of 
    sigmas from the mean of the array """
    mean = np.median(x)
    std = np.std(x)
    z = np.abs(x - mean)/std
    mask = (z < sigma)
    return mask


def filter_z_test(shifts, set_to_nan=False):
    """ return filtered shifts collection by z test of sigma 5 """
    
    # Filter
    df = pd.DataFrame(shifts, copy=True)
    df.columns = ["x", "err", "valid"]

    mask = compute_z_test_mask(df["x"], 5)

    if set_to_nan:
        df.valid[mask == False] = np.nan 
    else:
        df.valid[mask == False] = False 

    return np.asarray(df)



def filter_z_test_result(result, set_to_nan=False):
    """ Takes in the "result" object from the compute_all_feature_shifts and
    applies the z test filter. """
    filtered_result = []
    for file_result in result:
        s = filter_z_test(file_result, set_to_nan=set_to_nan)
        filtered_result.append(s)
    filtered_result = np.asarray(filtered_result, dtype=object)
    return filtered_result


def matrix_reduce_results_file(filename, path, plot=True, with_date_duplicates=False):
    """ Takes a file of our cross-correlation results (matrix) and reduces"""

    result, coords = np.load(filename, allow_pickle=True)
    diff_matrix, diff_matrix_err, diff_matrix_valid = parse_matrix_results(result, coords)
    return matrix_reduce(diff_matrix, diff_matrix_err, diff_matrix_valid, path, plot, with_date_duplicates)


def matrix_reduce(diff_matrix, diff_matrix_err, diff_matrix_valid, path, plot=True, with_date_duplicates=False):
    """ Takes a file of our cross-correlation results (matrix) and reduces"""

    def model_chi2(*V):
        V = np.asarray([*V])
        res = []
        size = diff_matrix.shape[0] 
        for x in np.arange(size):
            for y in np.arange(x, size):
                if x != y:
                    res.append(((diff_matrix[x, y] - (V[x] - V[y])) / diff_matrix_err[x, y])**2)
        chi2 = np.sum(res)
        return chi2
    model_chi2.errordef = 1

    # Use zeros as init values
    init_values = np.zeros(diff_matrix.shape[0])

    minuit = Minuit(model_chi2, *init_values)
    m = minuit.migrad()
    final_shifts = minuit.values[:]
    final_shifts_err = minuit.errors[:]

    # Center around 0
    final_shifts = final_shifts - np.mean([min(final_shifts), max(final_shifts)])

    # get list of observation days
    if with_date_duplicates:
        dates = get_spectra_dates(get_all_spectra_filenames(path))
    else:
        dates = get_spectra_dates(get_spectra_filenames_without_duplicate_dates(path))

    days = convert_dates_to_relative_days(dates)

    if plot == False:
        return m, final_shifts, final_shifts_err, days

    # Plot: 
    velocity_shifts = get_above_diagonal(diff_matrix)
    velocity_shifts_err = get_above_diagonal(diff_matrix_err)

    # Center around zero:
    velocity_shifts = velocity_shifts - np.mean([min(velocity_shifts), max(velocity_shifts)])

    print(len(days), len(velocity_shifts))

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 3))
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Velocity shift [m/s]")
    ax1.set_title("Above diagonal")
    # Plot above-diagonal (days - 1 lenght because above diagonal is one shorter)
    ax1.errorbar(days[:-1], velocity_shifts, yerr=velocity_shifts_err, fmt=".", color="k", ms=1, elinewidth=0.5)
    ax1.set_ylim(min(velocity_shifts)*1.1, max(velocity_shifts) * 1.35)
    text = f"mean error = {np.mean(velocity_shifts_err):.3} m/s, rms = {(compute_rms(velocity_shifts)):.3} m/s"
    ax1.text(0.05, 0.95, text,
                    size = 8,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax1.transAxes)

    # Plot matrix reduction results
    ax2.set_xlabel("Time [days]")
    ax2.set_title("Matrix chi2 reduction")
    ax2.errorbar(days, final_shifts, yerr=final_shifts_err, fmt=".", color="k", ms=1, elinewidth=0.5)
    ax2.set_ylim(min(final_shifts)*1.1, max(final_shifts) * 1.35)
    text = f"mean error = {np.mean(final_shifts_err):.3} m/s, rms = {(compute_rms(final_shifts)):.3} m/s"
    ax2.text(0.05, 0.95, text,
                    size = 8,
                    horizontalalignment='left',
                    verticalalignment='top',
                    transform=ax2.transAxes)

    fig.tight_layout()
    # fig.savefig("rooo.png", dpi=300)
    return m, final_shifts, final_shifts_err, days


def fit_final_shifts(rv, rv_err, diff_matrix, diff_matrix_err, with_date_duplicates=True, save_as=None, path=SPECTRA_PATH_34411):

    # Get times
    if with_date_duplicates:
        filenames = get_all_spectra_filenames(path)
    else:
        filenames = get_spectra_filenames_without_duplicate_dates(path)

    seconds = [get_spectra_seconds_since_epoch(file) for file in filenames]
    seconds = np.asarray(seconds)
    days = seconds / (60*60*24)
    days = days - min(days)
    fig, (ax1) = plt.subplots(ncols=1, figsize=(8, 3.5))

    # Get direct comparisons (above diagonal) for comparison
    velocity_shifts = get_above_diagonal(diff_matrix) * 1/1000 # km/s
    velocity_shifts_err = get_above_diagonal(diff_matrix_err) * 1/1000 # km/s

    # Plot above diagonal
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel("Relative RV [km/s]")
    ax1.errorbar(days[:-1], velocity_shifts, yerr=velocity_shifts_err, fmt=".", color="C0", label="Above diagonal")

    # Fit
    x = days
    y = np.asarray(rv[:]) * 1/1000 # km/s
    y_err = np.asarray(rv_err[:]) * 1/1000 # km/s
    ax1.errorbar(x, y, yerr=y_err, fmt=".", color="k", label="Matrix reduction")

    # Fitting functions:
    def func(x, a, b, c, d) :
        return a * np.cos(x * b + c) + d

    # ChiSquare fit model:
    def model_chi(a, b, c, d) :
        y_fit = func(x, a, b, c, d)
        chi2 = np.sum(((y - y_fit) / y_err)**2)
        return chi2
    model_chi.errordef = 1

    minuit = Minuit(model_chi, a=30000, b=0.01, c=3.5, d=1040)
    m = minuit.migrad()        
                        
    # Plot result
    xPeak = np.linspace(x[0], x[len(x)-1], 100)
    ax1.plot(xPeak, func(xPeak, *minuit.values[:]), '-r')

    Npoints = len(x)
    Nvar = 4                                        # Number of variables
    Ndof_fit = Npoints - Nvar                       # Number of degrees of freedom = Number of data points - Number of variables
    Chi2_fit = minuit.fval                          # The chi2 value
    Prob_fit = stats.chi2.sf(Chi2_fit, Ndof_fit)    # The chi2 probability given N degrees of freedom

    a, a_err = -minuit.values['a'], minuit.errors['a']
            #  ^ A comes out negative for some reason, fix... 
    b, b_err = minuit.values['b'], minuit.errors['b']
    c, c_err = minuit.values['c'], minuit.errors['c']
    d, d_err = minuit.values['d'], minuit.errors['d']
    wavel, wavel_err = (2 * np.pi)/b, np.sqrt((2 * np.pi)**2*b_err**2/b**4)

    d = {'A = ':   [a, f"{a_err:.1}"],
        'b = ':    [b, f"{b_err:.1}"],
        'c = ':    [c, f"{c_err:.1}"],
        'd = ':    [d, f"{d_err:.1}"],
        'λ = ':    [wavel, f"{wavel_err:.1}"],
        'χ2 = ':     f"{Chi2_fit:.3}",
        'ndf = ':      Ndof_fit,
        'Prob = ':     Prob_fit,
        'f(x) = ':     "A cos(bx + c) + d"
    }

    matplotlib.rcParams["text.usetex"] = False
    text = nice_string_output(d, extra_spacing=2, decimals=3)
    add_text_to_ax(0.66, 0.95, text, ax1, fontsize=9, color='black')
    matplotlib.rcParams["text.usetex"] = True
    ax1.legend(loc = "upper left")

    if save_as is not None:
        fig.savefig(save_as, bbox_inches="tight", dpi=300)
        



def plot_matches(matches, valid_matches_mask=None, ncols=10, nrows=-1, return_fig=False):
    """ Plots matches """

    if nrows == -1:
        nrows = 1
        while nrows * ncols < len(matches):
            nrows += 1

    assert nrows < 100, "Height is higher than 100"

    # Plot matches
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 2, nrows * 1.5))
    for k in np.arange(len(matches)):
        match = matches[k]

        if k >= len(axs.flat):
            break

        ax = axs.flat[k]
        plot_match(match, ax)

        # if np.isclose(np.sum(spec1), np.sum(spec2), 0.1) == False:
        #     ax.set_facecolor('pink')

        if valid_matches_mask is not None and (valid_matches_mask[k] == False):
            ax.set_facecolor('pink')

    fig.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.05)

    # Remove ticks
    for ax in axs.flat:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    if return_fig:
        return fig


def plot_match(match, ax=None):
    matplotlib.rcParams["text.usetex"] = False

    if ax is None:
        fig, ax = plt.subplots(figsize=(6,3))
    
    f1 = match[0]
    f2 = match[1]
    wavel1 = f1[0]
    spec1 = f1[1]
    peak1 = f1[3]
    wavel2 = f2[0]
    spec2 = f2[1]
    peak2 = f2[3]

    # Plot
    ax.plot(wavel1, spec1, color="C0")
    ax.vlines(peak1, 0, 1, linestyle="dashed", color="C0")
    ax.plot(wavel2, spec2, color="C3")
    ax.vlines(peak2, 0, 1, linestyle="dashed", color="C3")

    # Plot peak difference
    diff = peak1 - peak2
    area_diff = np.abs(np.sum(spec1) - np.sum(spec2))
    text = f"Δλ = {diff:.3f} Å \nΔA = {area_diff:.3f}"
    ax.text(0.02, -0.2, text, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    matplotlib.rcParams["text.usetex"] = True


def plot_compare_lily(my_rv, my_err, days, save_as = None, star_name=None, 
            plot_residuals=False, lily_data_file=None, padding_top=4, padding_bottom=1, ticks=None, small_height=False):

    height = 4 if small_height else 5
    info_text_y_loc = 0.93 if small_height else 0.95

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(8, height), sharex=True, gridspec_kw={'height_ratios': [3, 3, 1.5]})

    # Load Lily data:
    df = pd.read_csv("lily_stellar_activity_data/" + lily_data_file)
    rv = df["CBC RV [m/s]"]
    rv_err = df["CBC RV Err. [m/s]"]

    # Find range
    ymin = floor(min([min(rv), min(my_rv)])) - padding_bottom
    ymax = ceil(max([max(rv), max(my_rv)])) + padding_top
    print(ymin, ymax)

    if ticks:
        ax1.yaxis.set_ticks(ticks)
        ax2.yaxis.set_ticks(ticks)

    ax1.set_ylabel("RV shift [m/s]")
    ax1.errorbar(days, my_rv, yerr=my_err, fmt=".", color="k", ms=1, elinewidth=0.5, label="My method")
    ax1.set_ylim(ymin, ymax)
    ax1.set_xlim(-20, max(days) + 20)
    ax1.hlines(y=0, xmin=-20, xmax=max(days) + 20, color="black", alpha=0.25, lw=0.1)
    ax1.legend(loc="upper left", prop={'size': 8})
    text = f"rms = {(compute_rms(my_rv)):.3} m/s, mean error = {np.mean(my_err):.2} m/s"
    ax1.text(0.19, info_text_y_loc, text,
                        size = 8,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax1.transAxes)

    if star_name:
        ax1.text(0.99, info_text_y_loc, star_name,
                        size = 8,
                        horizontalalignment='right',
                        verticalalignment='top',
                        transform=ax1.transAxes)


    
    t = df["Time [MJD]"]
    t = t - min(t)
    rms2 = compute_rms(rv)

    ax2.errorbar(t, rv, yerr=rv_err, fmt=".", ms=1, color="k", elinewidth=0.5, label="Lily Zhao et al's method")
    ax2.set_ylabel("RV shift [m/s]")
    ax2.set_ylim(ymin, ymax)
    ax2.set_xlim(-20, max(t) + 20)
    ax2.hlines(y=0, xmin=-20, xmax=max(t) + 20, color="black", alpha=0.25, lw=0.1)
    ax2.legend(loc="upper left", prop={'size': 8})

    text = f"rms = {(rms2):.3} m/s, mean error = {np.mean(rv_err):.2f} m/s"
    ax2.text(0.30, info_text_y_loc, text,
                        size = 8,
                        horizontalalignment='left',
                        verticalalignment='top',
                        transform=ax2.transAxes)

    # ax1.get_shared_x_axes().join(ax1, ax2)

    # ax1.sharex(ax3)
    # ax2.sharex(ax3)
    fig.subplots_adjust(wspace=0, hspace=0)
    # fig.tight_layout()

    # plot residuals
    residuals = my_rv - rv
    ax3.scatter(t, residuals, s=0.5, color="k")
    ax3.set_xlabel("Time [days]")
    ax3.set_ylabel("Residuals")
    ax3.yaxis.set_ticks([-2, 0, 2])
    ax3.hist(residuals, orientation='horizontal', range=(-2.5, 2.5), bins=int(np.sqrt(len(residuals))), alpha=0.1, color="k", bottom=-20)


    if save_as is not None:
        fig.savefig(save_as, bbox_inches="tight", dpi=300)

    # Histogram of residuals if same length
    if plot_residuals:
        if len(rv) == len(my_rv):

            fig, ax = plt.subplots(figsize=(6,0.5))
            ax.plot(t, my_rv - rv, ".", ms=0.5)
            ax.set_ylabel("Residuals [m/s]")
            ax.set_xlabel("Days")

            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6,1.5))
            ax1.hist(my_rv - rv, bins=50, range=(-5, 5));
            ax2.scatter(my_rv, rv, s=0.5)



# ================================================================================================
# 
#                                   Analysing order by order
# 
# ================================================================================================


def prepare_orders(filename, max_frac_err = 0.1, is_peg_51 = False):
    """ Prepares data in a dataframe

    Data-filtering:
        - Data points outside EXCALIBUR_MASK are excluded.
        - Orders with less than 10 data points are excluded.
        - data points with large fractional error are excluded.

    Args:
        filename (fits file): the file
        max_frac_err (float, optional): exclude data points with larger fractional error (spectrum / photon count). Defaults to 0.1.
        angstrom (bool, optional): True if you want wavelengths in angstrom and not cm/s. Defaults to False.
        is_peg_51 (bool, optional) : True if using data from Peg51, which has other column names. Defaults to False.

    Returns:
        pd.DataFrame: Return a pandas dataframe from a given fits file with the following data:
            x : bary_excalibur in cm/s
            y : spectrum
            y_err : spectrum error
            order : order 
    """

    order_data = []
    fits_data = load_spectra_fits(filename)
    orders_n = shape(fits_data["spectrum"])[0]

    orders = np.arange(0, orders_n)
    orders = np.arange(37, 76)
    for order in orders:
        
        if is_peg_51:
            pixel_mask  = fits_data['pixel_mask'][order]
            y           = fits_data['spectrum'][order][pixel_mask]
            y_err       = fits_data['uncertainty'][order][pixel_mask]
            continuum   = fits_data['continuum'][order][pixel_mask]
            x           = fits_data['bary_wavelength'][order][pixel_mask]
        else: 
            excalibur_mask  = fits_data['EXCALIBUR_MASK'][order]
            y           = fits_data['spectrum'][order][excalibur_mask]
            y_err       = fits_data['uncertainty'][order][excalibur_mask]
            continuum   = fits_data['continuum'][order][excalibur_mask]
            x           = fits_data['BARY_EXCALIBUR'][order][excalibur_mask]
        
        # skip order if no good data
        if len(x) < 10 or len(y) < 10:
            continue

        # Normalize intensity by continuum 
        y = y/continuum
        y_err = y_err/continuum

        # filter by fractional error 
        frac_err = y_err/y
        frac_err_mask = (0 < frac_err) & (frac_err < max_frac_err) # reject if larger than 10% and negative
        y = y[frac_err_mask]
        y_err = y_err[frac_err_mask]
        x = x[frac_err_mask]

        # Now invert peaks
        y = 1 - y

        # Create DataFrame and append
        data = {
            'x': x,
            'y': y,
            'y_err': y_err,
            'order': order
            }
        order_data.append(pd.DataFrame(data))
    
    # concatenate dataframes
    df = pd.concat(order_data)
    return df



def compute_order_shift(df1, df2, order=-1, plot=False, zoom_plot=True, ax=None):
    """Computes the wavelength shift for a given order between two files (dataframes)

    Args:
        df1 (pd.dataframe): dataframe with order data for file 1
        df2 (pd.dataframe): dataframe with order data for file 2
        order (int, optional): Order number, just for the plot, or if needed in the returned dataframe. Defaults to -1.
        plot (bool, optional): Plot order. Defaults to False.
        zoom_plot (bool, optional): Zoom in on plot. Defaults to True.
        ax (plt.ax, optional): optional ax to plot on. Defaults to None.

    Returns:
        pd.dataframe: Dataframe with computed shift_val, shift_err, minuit_valid, summed_diff, order
    """

    c = 299792458 # m/s
    
    # Interp first file
    f1 = interp1d(df1.x, df1.y, kind='cubic', fill_value="extrapolate")
    f1_upper_err = interp1d(df1.x, df1.y + df1.y_err, kind='cubic', fill_value="extrapolate")
    f1_lower_err = interp1d(df1.x, df1.y - df1.y_err, kind='cubic', fill_value="extrapolate")

    # ChiSquare fit model:
    def model_chi2(A):

        # Interpolate template
        # interp_df2 = df2.x + A
        interp_df2 = df2.x * (1 + A/c) # Wavelengths are be stretched by a factor of (1 + v/c)
        f2 = interp1d(interp_df2, df2.y, kind='cubic', fill_value="extrapolate")

        # Find common x-range
        xmin = max([min(df1.x), min(interp_df2)])
        xmax = min([max(df1.x), max(interp_df2)])
        xnewCommon = np.linspace(xmin, xmax, 1000)
        
        # Evaluate interpolation
        ynew1 = f1(xnewCommon)
        ynew2 = f2(xnewCommon)

        # Evalute error interpolation
        ynew1_upper_err = f1_upper_err(xnewCommon)
        ynew1_lower_err = f1_lower_err(xnewCommon)
        ynew1_upper_err_abs = np.abs(ynew1 - ynew1_upper_err)
        ynew1_lower_err_abs = np.abs(ynew1 - ynew1_lower_err)
        ynew1_err = np.mean([ynew1_upper_err_abs, ynew1_lower_err_abs], axis=0) # pairwise mean 

        # Compute chi2
        chi2 = np.sum(((ynew1 - ynew2) / ynew1_err)**2)
        return chi2
    model_chi2.errordef = 1
        
    # For feature approach we would take the diff between the peaks, but here
    # let's try to just use the wavelength median as a rough estimate.
    peak1 = np.median(df1.x)
    peak2 = np.median(df2.x)

    A_init = (peak1 / peak2 - 1 ) * c # shift between the two peaks

    # Compute bounds on A
    x1_min, x1_max = min(df1.x), max(df2.x)
    A_lower_bound = (x1_min / peak2 - 1 ) * c
    A_upper_bound = (x1_max / peak2 - 1 ) * c

    minuit = Minuit(model_chi2, A=A_init)
    minuit.limits["A"] = (A_lower_bound, A_upper_bound)
    minuit.migrad()

    # Results
    valid = minuit.valid
    shift_min_final = minuit.values['A']
    shift_min_final_err = minuit.errors['A']
    summed_diff = np.sum(df1.y - df2.y) # doesn't have much to do with the fit, but just the difference from night to night in counts
    forced = minuit.fmin.has_made_posdef_covar
    at_limit = minuit.fmin.has_parameters_at_limit

    if forced or at_limit:
        valid = False

    # Plot final shifted values
    if plot:

        if ax == None:
            fig, ax = plt.subplots(figsize=(14,6))

        ax.plot(df1.x, df1.y, ".", label=f"df1 ({order}. order)")
        ax.plot(df2.x, df2.y, ".",label="df2")
        ax.plot(df2.x + shift_min_final, df2.y, ".", label="Fit")
        ax.legend(loc="upper right")

        # Zoom:
        if zoom_plot:
            mid = np.median(f1.x)
            ax.set_xlim(mid - 2, mid + 2)
    

    # Create DataFrame and append
    data = {
        'shift_val':    [shift_min_final],
        'shift_err':    [shift_min_final_err],
        'minuit_valid': [valid],
        'summed_diff':  [summed_diff],
        'order':        [order]
        }
    return pd.DataFrame(data)


def compute_all_orders_shift(filename1, filename2, plot_overview=False, plot_orders=False, is_peg_51 = False):
    """Compute shift for all orders in two files.

    Args:
        filename1 (string path): fits file 1
        filename2 (string path): fits file 2
        plot_overview (bool, optional): plot the shifts for all orders in one plot. Defaults to False.
        plot_orders (bool, optional): plot all orders individually. Defaults to False.
        is_peg_51 (bool, optional) : True if using data from Peg51, which has other column names. Defaults to False.

    Returns:
        (float) : weighted mean for all orders
        (float) : weighted err for all orders
        (float) : ratio of valid minuit fits
    """
    

    f1 = prepare_orders(filename1, is_peg_51=is_peg_51)
    f2 = prepare_orders(filename2, is_peg_51=is_peg_51)

    # find common orders : interesection of lists of unique orders
    common_orders = np.intersect1d(np.unique(f1.order), np.unique(f2.order))

    results = []
    for o in common_orders:
        r = compute_order_shift(f1[f1.order == o], f2[f2.order == o], order=o, plot=plot_orders)    
        results.append(r)

    df = pd.concat(results)

    # plot
    if plot_overview:

        zmask = compute_z_test_mask(df.shift_val, 5)
        df_valid = df[zmask].copy()
        df_invalid = df[zmask == False].copy()

        shift, shift_err = weighted_mean(df_valid.shift_val, df_valid.shift_err)

        fig, ax1 = plt.subplots(figsize=(14, 10))
        ax1.errorbar(df_valid.shift_val, df_valid.order, xerr=df_valid.shift_err, fmt=".", linewidth=5, color = "C2")
        ax1.errorbar(df_invalid.shift_val, df_invalid.order, xerr=df_invalid.shift_err, fmt=".", linewidth=5, color = "C3")

        ax1.vlines(shift, min(df.order), max(df.order), linestyle="dashed", alpha=0.5, color="black", label=f"Weighted average = {shift:.3} ± {shift_err:.1}")
        ax1.axvspan(shift-shift_err, shift+shift_err, alpha=0.2) # too small to see anyway

        ax1.invert_yaxis()  # invert axis so feature 0 starts at the top
        ax1.set_title("Shift")
        ax1.set_xlabel("RV [cm/s]")
        ax1.set_ylabel("Order")
        ax1.legend()

    return df



# ================================================================================================
# 
#                                   Analysing chunks
# 
# ================================================================================================










# ================================================================================================
