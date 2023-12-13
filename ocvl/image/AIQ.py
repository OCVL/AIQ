import math
import matplotlib.pyplot as plt
import scipy
from skimage.transform import warp_polar
import numpy as np
import cv2
from tkinter import filedialog, simpledialog
from mpl_toolkits.axes_grid1 import make_axes_locatable


def hanning_2d(dimension, fraction):
    """
    Function that determines the Hanning window's size
    :param dimension: The image's dimensions
    :param fraction: The amount of overlap between the generated ROIs to be used in Welch's Method
    :return: The 2D Hanning window to be used
    """

    # Only use the smallest dimension -this forces a square ROI and simplifies some assumptions
    smallest_dim = np.min(dimension)
    han_width = np.hanning(np.floor(smallest_dim * fraction))  # This is the hanning window width
    han_height = np.hanning(np.floor(smallest_dim * fraction))  # This is the hanning window height

    # Have to use at least_2d to ensure that the transpose doesn't just return what array you started with..
    hann_2d = np.multiply(han_width, np.atleast_2d(han_height).T)  # view inputs as arrays with a least 2D

    return hann_2d


def welch_2d_windowing(image, window):
    """
    Function that performs Welch's method.
    :param image: The image's data that was inputted
    :param window: The Hanning window that was determined to be used for this image
    :return: The smooth power spectrum determined through Welch's method
    """

    im_dim = image.shape  # Shape parameter of the image
    win_dim = window.shape  # Shape parameter of the window

    # Create steps with a 50% overlap between the windows
    rows = range(0, int(im_dim[0] - win_dim[0]), int(np.floor(
        win_dim[0] / 2.0)))
    cols = range(0, int(im_dim[1] - win_dim[1]), int(np.floor(win_dim[1] / 2.0)))

    all_roi = np.zeros(shape=[win_dim[0], win_dim[1], (len(rows)) * (len(cols))])

    r = 0
    for i in range(len(rows)):
        for j in range(len(cols)):
            windowed_roi = np.multiply(window, image[rows[i]:(rows[i] + win_dim[0]), cols[j]:(cols[j] + win_dim[1])])

            comp_arr = scipy.fft.fftshift(scipy.fft.fft2(windowed_roi))

            # Normalize our dft by sqrt(N) so that window size doesn't affect SNR estimation.
            comp_arr = np.divide(comp_arr, np.sqrt(win_dim[0] * win_dim[1]))
            comp_arr = np.abs(comp_arr) * np.abs(comp_arr)

            all_roi[:, :, r] = comp_arr
            r += 1

    smooth_pwr_spect = np.mean(all_roi, axis=2)
    return smooth_pwr_spect


def calculate_snr(welch_pwr_spect):
    """
    Function that takes the welch power spectrum and converts it to a polar spectrum. Following derivatives and
    integration occur before application of the designed cutoff frequencies.
    :param welch_pwr_spect: The Welch power spectrum of the image
    :return: the final SNR value of the image
    """

    theta_sampling = 1
    rho_sampling = 0.5

    # Our maximum valid radius is only as big as half our window size.
    max_rad = int(np.floor(welch_pwr_spect.shape[0] / 2)) + 1

    polar_spect_welch = warp_polar(welch_pwr_spect, radius=max_rad, output_shape=(
        360 / theta_sampling, max_rad / rho_sampling))  # Change the coordinate space of the image
    polar_spect_welch = polar_spect_welch[3:177, :]

    # Make the 0's nan, so that when we calculate our mean we exclude the areas.
    polar_spect_welch[polar_spect_welch == 0] = np.nan

    # Determined to throw off the spectrum if these angles are included
    polar_spect_welch = np.delete(polar_spect_welch,
                                  [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93], axis=0)

    polar_avg_welch = np.nanmean(polar_spect_welch, axis=0)  # compute the mean along the specified axis, ignoring Nans
    bef = len(polar_avg_welch)
    aft = bef - 2
    polar_avg_welch = polar_avg_welch[0:aft]

    # Determine the frequency cutoff between signal/noise
    freq_bin_size_welch = rho_sampling / polar_spect_welch.shape[1]  # The size/span of each bin
    freq_bins_welch = np.arange(polar_spect_welch.shape[
                             1] - 2) * freq_bin_size_welch  # evenly space the values and multiply by the size/span of each bin
    freq_bins_welch[0] = -1  # Our first element is the DC term - so exclude it

    spacing_bins_welch = 1 / freq_bins_welch
    spacing_bins_welch[0] = 10000
    low_noise_cutoff_welch = 1 / (0.5 * 0.0575)
    high_noise_cutoff_welch = 1 / (0.5 * 0.7)

    freq_bins_welch[0] = 0

    # find the range that the noise is in
    signal_range_welch = polar_avg_welch[
        (spacing_bins_welch <= low_noise_cutoff_welch) & (spacing_bins_welch >= high_noise_cutoff_welch)]

    # this is the total range of frequencies in the image
    total_range_welch = polar_avg_welch[(spacing_bins_welch < high_noise_cutoff_welch)]

    signal_power_change_welch = freq_bin_size_welch * np.sum(abs(np.diff(signal_range_welch)))
    noise_power_change_welch = freq_bin_size_welch * np.sum(abs(np.diff(total_range_welch)))

    # Final calculation of the SNR value for the image
    snr = 10 * np.log10(signal_power_change_welch / noise_power_change_welch)

    return snr


def aiq(image):
    """
    Function that runs through the completed algorithm to provide the SNR value of the image inputted
    :param image: The image's data in an array format
    :return: The SNR value for the inputted image
    """

    # Create a window that is effectively a quarter of our image size.
    window_data = hanning_2d(image.shape, 0.25)  # makes the hanning window defined above

    welch_power = welch_2d_windowing(image, window_data)  # Calling the function that creates the graph window

    snr_value = calculate_snr(welch_power)

    return str(snr_value)


def load_image(image_filename):
    """
    Function that loads in the image selected as a gray scale image
    :param image_filename: Name of the image to be loaded in
    :return: The image's data in an array format
    """

    # Allow the user to select the image that is to be used in the analysis
    if image_filename:
        im = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)

        return im


def get_files():
    """
    Function that allows the user to select all the image files that are to be ran through the algorithm
    :return: List of files that were selected
    """

    f2 = filedialog.askopenfilenames()
    file_list = list(f2)
    return file_list


if __name__ == '__main__':
    # Get all the images
    images = get_files()
    fileHandle = open("SNR_values.txt", "w")

    for i in images:
        # Load in image
        imageData = load_image(i)

        # Calculate the SNR value and write it to file
        whole = aiq(imageData)
        fileHandle.write(str(i) +"," + str(whole) + "\n")

    fileHandle.close()

