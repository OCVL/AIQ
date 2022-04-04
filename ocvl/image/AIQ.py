import math
import matplotlib.pyplot as plt
import scipy
from skimage.transform import warp_polar
import numpy as np
import cv2
import csv
from tkinter import filedialog, simpledialog


def hanning_2d(dimension, fraction):
    # Only use the smallest dimension- this forces a square ROI and simplifies some assumptions
    smallestdim = np.min(dimension)
    han_width = np.hanning(np.floor(smallestdim * fraction))  # This is the hanning window width - Its a bell shape
    han_height = np.hanning(np.floor(smallestdim * fraction))  # This is the hanning window height - Its a bell shape

    # Have to use atleast_2d to ensure that the transpose doesn't just return what array you started with..
    hann_2d = np.multiply(han_width, np.atleast_2d(han_height).T)  # view inputs as arrays with a least 2D

    return hann_2d


def welch_2d_windowing(image, window):
    imDim = image.shape  # Shape parameter of the image
    winDim = window.shape  # Shape parameter of the window

    # Create steps with a 50% overlap between the windows
    rows = range(0, int(imDim[0] - winDim[0]), int(np.floor(
        winDim[0] / 2.0)))  # Starts at 0, stop at imdim[0]-winDim[0] and increase by the 3rd parameter
    cols = range(0, int(imDim[1] - winDim[1]), int(np.floor(winDim[1] / 2.0)))

    allroi = np.zeros(shape=[winDim[0], winDim[1], (len(rows)) * (len(cols))])  # array of zeros with the shape defined

    # This is just setting up the window for the graph I believe
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
    stddev_pwrspect = np.std(allroi, axis=2)  # This is not used why is that?

    # plt.imshow(np.log10(stddev_pwrspect), cmap="gray")
    # plt.show()
    return smooth_pwrspect


def aiq(image):
    # Create a window that is effectively a quarter of our image size.
    windowData = hanning_2d(image.shape, 0.25)  # makes the hanning window defined above

    welch_pwr_spect = welch_2d_windowing(image, windowData)  # Calling the function that creates the graph window

    thetasampling = 1
    rhosampling = 0.5

    # Our maximum valid radius is only as big as half our window size.
    maxrad = int(np.floor(welch_pwr_spect.shape[0] / 2)) + 1

    polar_spect = warp_polar(welch_pwr_spect, radius=maxrad, output_shape=(
        360 / thetasampling, maxrad / rhosampling))  # Change the coordinate space of the image
    # Remove the last column because it always seems to be all zeros anyway.
    polar_spect = polar_spect[0:180, :]

    #plt.imshow(np.log10(polar_spect), cmap="gray")
    #plt.title("Polar Plot")
    #plt.show()

    # Make the 0's nan, so that when we calculate our mean we exlcude the exluded areas.
    polar_spect[polar_spect == 0] = np.nan

    polar_avg = np.nanmean(polar_spect, axis=0)  # compute the mean along the specified axis, ignoring Nans
    bef = len(polar_avg)
    aft = bef - 2
    polar_avg = polar_avg[0:aft]

    # Determine the frequency cutoff between signal/noise
    freq_bin_size = rhosampling / polar_spect.shape[1]  # The size/span of each bin
    freqBins = np.arange(polar_spect.shape[
                             1] - 2) * freq_bin_size  # evenly space the values in the array and multiply them by the size/span of each bin
    freqBins[0] = -1  # Our first element is the DC term and won't be included in the calculation anyway- so exlcude it

    spacing_bins = 1 / freqBins
    spacing_bins[0] = 10000
    low_noise_cutoff = 1 / (0.5 * 0.0575)
    high_noise_cutoff = 1 / (0.5 * 0.7)

    freqBins[0] = 0
    '''
    plt.plot(freqBins, np.log10(polar_avg), color='c')
    plt.title("Polar Plot")
    plt.show()
    sname = "D:\Brea_Brennan\Image_Quality_Analysis\Poster Materials\PolarPlot_ARVOPoster.svg"
    plt.savefig(sname)
    plt.clf()
    '''

    signal_range = polar_avg[
        (spacing_bins <= low_noise_cutoff) & (spacing_bins >= high_noise_cutoff)]  # find the range that the noise is in
    total_range = polar_avg[(spacing_bins < high_noise_cutoff)]  # this is the total range of frequencies in the image

    # Use of the DERIVATIVE of the power spectrum captures any areas of interest that would cause sudden changes
    diffs = np.diff(polar_avg)

    signal_power_change = freq_bin_size * np.sum(abs(np.diff(signal_range)))
    noise_power_change = freq_bin_size * np.sum(abs(np.diff(total_range)))

    '''
    # The code for the derivative graph should go here
    xs = np.linspace(0, (len(diffs) - 1), len(polar_avg) - 1)
    plt.plot(xs, diffs, color='y', label='derivative')
    plt.title("Derivative Plot")
    plt.show()
    dname = "D:\Brea_Brennan\Image_Quality_Analysis\Poster Materials\derivativePlot_ARVOPoster.svg"
    plt.savefig(dname)
    plt.clf()
    '''

    SNR = 10 * np.log10(signal_power_change / noise_power_change)
    # print("Estimated SNR of this image is: " + str(SNR))

    return str(SNR)


def load_image(image_filename):
    # Allow the user to select the image that is too be used in the analysis
    if image_filename:
        image = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)

    return image


def get_files():
    f2 = filedialog.askopenfilenames()
    fileList = list(f2)
    return fileList


def ROIExtraction(data):
    dim = 128  # Starting with 128x128 pixels
    imsize = data.shape
    rows = imsize[0]
    cols = imsize[1]
    ovl = .25  # over lap between the ROIs 1- % you want
    inc = int(np.ceil(dim * ovl))

    # Generate all the arrays and lists needed to operate
    ROIs = np.empty((dim, dim), dtype=float)
    SNR_sum = np.empty((rows, cols), dtype=float)
    sum_map = np.empty((rows, cols), dtype=float)
    avg_SNR_map = np.empty((rows, cols), dtype=float)
    outVals = []

    # Set all values in the arrays to start at zero
    SNR_sum[:, :] = 0.0
    sum_map[:, :] = 0.0
    avg_SNR_map[:, :] = 0

    # Loop for making the ROIs - ignores the boarders on right and bottom that does not give full ROI of dim by dim
    for y in range(0, rows-dim, inc):
        for x in range(0, cols-dim, inc):

            ROIs = np.dstack((ROIs, data[y:(y + dim), x:(x + dim)]))
            val = aiq(data[y:(y + dim), x:(x + dim)])  # get the SNR Value of the ROI
            outVals.append(val) # add the value to a list of all the SNR values

            # load the SNR val into the ROI dimensions for the image sum
            SNR_sum[y:(y+dim), x:(x+dim)] += float(val)
            sum_map[y:(y + dim), x:(x + dim)] += 1  # add one to the sum map

    ROIs = ROIs[:, :, 1:]   # Remove the first ROI with how the variable was generated

    # Generate the average SNR map from the two sum maps
    avg_SNR_map[0:(y+dim), 0:(x+dim)] = np.divide(SNR_sum[0:(y+dim), 0:(x+dim)], sum_map[0:(y+dim), 0:(x+dim)])
    avg_SNR_map[avg_SNR_map == 0] = np.nan

    return ROIs, avg_SNR_map, SNR_sum, sum_map, outVals


if __name__ == '__main__':
    # Get all the images
    print("Start\n")
    images = get_files()
    w = 0  # set to 1 in order to write the ROI
    SNRs = []
    sumR = 0
    q = 0  # set to 1 for ROI implementation
    imNums = 1  # track the image numbers for naming purposes

    if q != 1:
        fWhole = open("AOIP_Confocal_Updated_SNR_Whole.txt", "w")
    else:
        fROI = open("SNR_MAP_ROI_64_75p_AC3.txt", "w")  # file to save the results in

    for i in images:
        imageData = load_image(i)
        if q != 1:
            whole = aiq(imageData)
            print("whole image SNR: " + str(whole))
            fWhole.write(str(whole) + "\n")
            imNums += 1
        # If an ROI is needed q will be 1
        else:
            R, Gen_map, SNR_map, track_map, SNRs = ROIExtraction(imageData)
            print("Generated the maps successfully")
            # for d in range(SNRs.shape):
                # fROI.write(str(SNRs[d]) + "\n")
                # sumR = sumR + float(SNRs[d])

            n = len(SNRs)
            avgROIs = sumR / n
            '''
            # Save the generated arrays to construct the SNR maps in Matlab
            name = "PythonSNRGeneratedMap_64AC1.tif"
            savePath = "D:\Brea_Brennan\Image_Quality_Analysis\ROIs For Map\\" + name
            cv2.imwrite(savePath, SNR_map)

            name = "PythonSumGeneratedMap_64AC1.tif"
            savePath = "D:\Brea_Brennan\Image_Quality_Analysis\ROIs For Map\\" + name
            cv2.imwrite(savePath, track_map)
            '''
            name = "PythonAvgGeneratedMap_128MS3.tif"
            savePath = "D:\Brea_Brennan\Image_Quality_Analysis\ROIs For Map\\" + name
            cv2.imwrite(savePath, Gen_map)

        # Clear the entire list in case there are less ROIs generated in the next image and restart the sum
        SNRs.clear()
        sumR = 0

        # To write the ROI generated to there own tif files
        if w == 1:
            for t in range(R.shape[2]):
                name = "ROI" + str(t) + ".tif"
                savePath = "D:\Brea_Brennan\Image_Quality_Analysis\ROIs For Map\\" + name
                temp = R[..., t]
                if np.all(temp == 0):
                    print("All zeros! Do not right this ROI!")
                else:
                    cv2.imwrite(savePath, temp)

    if q != 1:
        fWhole.close()
    else:
        fROI.close()

    print("End\n")
