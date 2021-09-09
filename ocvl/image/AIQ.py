import matplotlib.pyplot as plt
import scipy
from skimage.transform import warp_polar
import numpy as np
import cv2
from tkinter import filedialog, simpledialog


def hanning_2d(dimension, fraction):
    # Only use the smallest dimension- this forces a square ROI and simplifies some assumptions
    smallestdim = np.min(dimension)
    han_width = np.hanning(np.floor(smallestdim * fraction))
    han_height = np.hanning(np.floor(smallestdim * fraction))

    # Have to use atleast_2d to ensure that the transpose doesn't just return what array you started with..
    hann_2d = np.multiply(han_width, np.atleast_2d(han_height).T)

    return hann_2d


def welch_2d_windowing(image, window):
    imDim = image.shape
    winDim = window.shape
    # Create steps with a 50% overlap between the windows
    rows = range(0, int(imDim[0] - winDim[0]), int(np.floor(winDim[0] / 2.0)))
    cols = range(0, int(imDim[1] - winDim[1]), int(np.floor(winDim[1] / 2.0)))

    allroi = np.zeros(shape=[winDim[0], winDim[1], (len(rows)) * (len(cols))])

    r = 0
    for i in range(len(rows)):
        for j in range(len(cols)):
            windowed_roi = np.multiply(window, image[rows[i]:(rows[i] + winDim[0]), cols[j]:(cols[j] + winDim[1])])
            comparr = scipy.fft.fftshift(scipy.fft.fft2(windowed_roi))
            # Normalize our dft by sqrt(N) so that window size doesn't affect SNR estimation.
            comparr = np.divide(comparr, np.sqrt(winDim[0] * winDim[1]))
            comparr = np.abs(comparr) * np.abs(comparr)

            allroi[:, :, r] = comparr
            r += 1

    smooth_pwrspect = np.mean(allroi, axis=2)
    stddev_pwrspect = np.std(allroi, axis=2)

    # plt.imshow(np.log10(stddev_pwrspect), cmap="gray")
    # plt.show()
    return smooth_pwrspect


def aiq(image):
    # Create a window that is effectively a quarter of our image size.
    windowData = hanning_2d(imageData.shape, 0.25)

    welch_pwr_spect = welch_2d_windowing(imageData, windowData)

    thetasampling = 1
    rhosampling = 0.5
    # Our maximum valid radius is only as big as half our window size.
    maxrad = int(np.floor(welch_pwr_spect.shape[0] / 2)) + 1

    polar_spect = warp_polar(welch_pwr_spect, radius=maxrad, output_shape=(360 / thetasampling, maxrad / rhosampling))
    # Remove the last column because it always seems to be all zeros anyway.
    polar_spect = polar_spect[0:180, :]

    # plt.imshow(np.log10(polar_spect), cmap="gray")
    # plt.show()

    # Make the 0's nan, so that when we calculate our mean we exlcude the exluded areas.
    polar_spect[polar_spect == 0] = np.nan

    polar_avg = np.nanmean(polar_spect, axis=0)

    # Determine the frequency cutoff between signal/noise
    freq_bin_size = rhosampling / polar_spect.shape[1]
    freqBins = np.arange(polar_spect.shape[1]) * freq_bin_size
    freqBins[0] = -1  # Our first element is the DC term and won't be included in the calculation anyway- so exlcude it

    scaleval = simpledialog.askfloat("Input", "What is the scale of the image in microns?")

    if scaleval is None:
        print("No scale value entered. Defaulting to 0.45 microns...")
        scaleval = 0.45

    spacing_bins = 1 / freqBins


    low_noise_cutoff = 1 / (0.5 * 0.045)
    high_noise_cutoff = 1 / (0.5 * 0.7)

    noise_range = polar_avg[(spacing_bins > 0) & (spacing_bins <= high_noise_cutoff)]
    total_range = polar_avg[(spacing_bins > high_noise_cutoff) & (spacing_bins <= low_noise_cutoff)]

    # Use of the DERIVATIVE of the power spectrum captures any areas of interest that would cause sudden changes
    tot_power_change = abs(freq_bin_size * np.sum(np.diff(noise_range)))
    noise_power_change = abs(freq_bin_size * np.sum(np.diff(total_range)))

    SNR = -10 * np.log10(tot_power_change / noise_power_change)
    print("Estimated SNR of this image is: " + str(SNR))

    freqBins[0] = 0
    plt.plot(freqBins, np.log10(polar_avg))
    plt.show()

def load_image():
    image_filename = filedialog.askopenfilename()
    if image_filename:
        image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)

        return image


if __name__ == '__main__':
    imageData = load_image()
    aiq(imageData)

