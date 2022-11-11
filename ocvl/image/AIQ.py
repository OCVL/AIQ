import math
import matplotlib.pyplot as plt
import scipy
from skimage.transform import warp_polar
import numpy as np
import cv2
from tkinter import filedialog, simpledialog
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
            im_win = image[rows[i]:(rows[i] + winDim[0]), cols[j]:(cols[j] + winDim[1])]
            '''
            # Step 2: Display the Image inside the window
            fig2 = plt.figure(2)
            plt.imshow(im_win, cmap='gray')
            plt.title("Image inside the Window")
            
            # Step 3: Display Windowed Image
            fig3 = plt.figure(3)
            plt.imshow(windowed_roi, cmap='gray')
            plt.title("Windowed Image")
            '''
            comparr = scipy.fft.fftshift(scipy.fft.fft2(windowed_roi))

            # Normalize our dft by sqrt(N) so that window size doesn't affect SNR estimation.
            comparr = np.divide(comparr, np.sqrt(winDim[0] * winDim[1]))
            comparr = np.abs(comparr) * np.abs(comparr)
            '''
            # Step 4: Display the DFT of the windowed image
            fig4 = plt.figure(4)
            plt.imshow(np.log10(comparr), cmap='gray')
            plt.title("DFT of Windowed Image")

            plt.show(block=False)
            '''
            allroi[:, :, r] = comparr
            r += 1

    smooth_pwrspect = np.mean(allroi, axis=2)
    # stddev_pwrspect = np.std(allroi, axis=2)  # This is not used why is that?


    # Step 5: Display the average of the DFT
    # fig5 = plt.figure(2)
    # plt.imshow(np.log10(smooth_pwrspect), cmap='gray')
    # plt.title("Average of the DFT from all Windowed Images")
    # plt.show(block=False)

    # plt.imshow(np.log10(stddev_pwrspect), cmap="gray")
    # plt.show()
    return smooth_pwrspect, allroi


def calculateSNR(welch_pwr_spect, totalROIS, image, window):
    thetasampling = 1
    rhosampling = 0.5

    # Our maximum valid radius is only as big as half our window size.
    maxrad = int(np.floor(welch_pwr_spect.shape[0] / 2)) + 1

    polar_spect_welch = warp_polar(welch_pwr_spect, radius=maxrad, output_shape=(
        360 / thetasampling, maxrad / rhosampling))  # Change the coordinate space of the image
    # Remove the last column because it always seems to be all zeros anyway.
    # polar_spect_welch = polar_spect_welch[0:180, :]
    polar_spect_welch = polar_spect_welch[3:177, :]

    # Make the 0's nan, so that when we calculate our mean we exlcude the exluded areas.
    polar_spect_welch[polar_spect_welch == 0] = np.nan

    polar_spect_welch = np.delete(polar_spect_welch,
                                  [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93], axis=0)

    # plt.figure(3)
    # plt.plot(np.log10(polar_spect_welch.transpose()))
    # plt.title("Polar Spect Welch Each row")
    # plt.show(block=False)


    polar_avg_welch = np.nanmean(polar_spect_welch, axis=0)  # compute the mean along the specified axis, ignoring Nans
    bef = len(polar_avg_welch)
    aft = bef - 2
    polar_avg_welch = polar_avg_welch[0:aft]

    # Determine the frequency cutoff between signal/noise
    freq_bin_size_welch = rhosampling / polar_spect_welch.shape[1]  # The size/span of each bin
    freqBins_welch = np.arange(polar_spect_welch.shape[
                             1] - 2) * freq_bin_size_welch  # evenly space the values in the array and multiply them by the size/span of each bin
    freqBins_welch[0] = -1  # Our first element is the DC term and won't be included in the calculation anyway- so exlcude it

    spacing_bins_welch = 1 / freqBins_welch
    spacing_bins_welch[0] = 10000
    low_noise_cutoff_welch = 1 / (0.5 * 0.0575)
    high_noise_cutoff_welch = 1 / (0.5 * 0.7)

    freqBins_welch[0] = 0

    '''
    # Step 6: Display the Average polar plot
    fig6 = plt.figure(6)
    plt.plot(freqBins, np.log10(polar_avg), color='c')
    plt.title("Polar Plot")
    '''

    # find the range that the noise is in
    signal_range_welch = polar_avg_welch[
        (spacing_bins_welch <= low_noise_cutoff_welch) & (spacing_bins_welch >= high_noise_cutoff_welch)]

    # this is the total range of frequencies in the image
    total_range_welch = polar_avg_welch[(spacing_bins_welch < high_noise_cutoff_welch)]

    # Use of the DERIVATIVE of the power spectrum captures any areas of interest that would cause sudden changes
    # diffs = np.diff(polar_avg)

    signal_power_change_welch = freq_bin_size_welch * np.sum(abs(np.diff(signal_range_welch)))
    noise_power_change_welch = freq_bin_size_welch * np.sum(abs(np.diff(total_range_welch)))

    '''
    # Step 7: Take the derivative of the polar average
    fig7 = plt.figure(7)
    xs = np.linspace(1/low_noise_cutoff, 1/high_noise_cutoff, len(diffs) - 1)
    plt.plot(freqBins[1:], diffs, color='r', label='derivative')
    # plt.xlim([(1/low_noise_cutoff)-0.005, (1/high_noise_cutoff)+0.005])
    plt.ylim([-200, 200])
    plt.axvline(1/low_noise_cutoff, color='c', label='Low Cutoff')
    plt.axvline(1/high_noise_cutoff, color='y', label='High Cutoff')
    plt.title("Derivative of Polar Plot")

    plt.show(block=False)
    '''

    SNR_welch = 10 * np.log10(signal_power_change_welch / noise_power_change_welch)
    SNR_ROIs = []

    # Perform the same calculations as above but for each of the ROIs
    for b in range(0, totalROIS.shape[2]):
        # Calculating the Polar spect for each ROI
        polar_spect = warp_polar(totalROIS[:, :, b], radius=maxrad, output_shape=(
            360 / thetasampling, maxrad / rhosampling))  # Change the coordinate space of the image

        # Remove the last column because it always seems to be all zeros anyway.
        polar_spect = polar_spect[0:180, :]

        # Make the 0's nan, so that when we calculate our mean we exlcude the exluded areas.
        polar_spect[polar_spect == 0] = np.nan

        polar_avg = np.nanmean(polar_spect, axis=0)  # compute the mean along the specified axis, ignoring Nans
        bef = len(polar_avg)
        aft = bef - 2
        polar_avg = polar_avg[0:aft]

        # Determine the frequency cutoff between signal/noise
        freq_bin_size = rhosampling / polar_spect.shape[1]  # The size/span of each bin

        # evenly space the values in the array and multiply them by the size/span of each bin
        freqBins = np.arange(polar_spect.shape[1] - 2) * freq_bin_size

        # Our first element is the DC term and won't be included in the calculation anyway- so exlcude it
        freqBins[0] = -1

        spacing_bins = 1 / freqBins
        spacing_bins[0] = 10000
        low_noise_cutoff = 1 / (0.5 * 0.0575)
        high_noise_cutoff = 1 / (0.5 * 0.7)

        freqBins[0] = 0

        # find the range that the noise is in
        signal_range = polar_avg[
            (spacing_bins <= low_noise_cutoff) & (spacing_bins >= high_noise_cutoff)]

        # this is the total range of frequencies in the image
        total_range = polar_avg[(spacing_bins < high_noise_cutoff)]

        signal_power_change = freq_bin_size_welch * np.sum(abs(np.diff(signal_range)))
        noise_power_change = freq_bin_size_welch * np.sum(abs(np.diff(total_range)))

        SNR = 10 * np.log10(signal_power_change / noise_power_change)

        SNR_ROIs.append(SNR)

    ''' HEAT MAP GENERATION CODE '''
    # Dimensions for heat map set up
    imDim = image.shape  # Shape parameter of the image
    winDim = window.shape  # Shape parameter of the window

    # Variable to hold the generated heat map
    heatMap = np.zeros(shape=image.shape)
    sum_map = np.zeros(shape=image.shape)

    # Create steps with a 50% overlap between the windows
    rows = range(0, int(imDim[0] - winDim[0]), int(np.floor(
        winDim[0] / 2.0)))  # Starts at 0, stop at imdim[0]-winDim[0] and increase by the 3rd parameter
    cols = range(0, int(imDim[1] - winDim[1]), int(np.floor(winDim[1] / 2.0)))

    count = 0
    for a in range(len(rows)):
        for b in range(len(cols)):
            # Add the SNR calculated above to the heat map
            heatMap[rows[a]:(rows[a] + winDim[0]), cols[b]:(cols[b] + winDim[1])] = heatMap[rows[a]:(rows[a] + winDim[0]), cols[b]:(cols[b] + winDim[1])] + SNR_ROIs[count]
            sum_map[rows[a]:(rows[a] + winDim[0]), cols[b]:(cols[b] + winDim[1])] = sum_map[rows[a]:(rows[a] + winDim[0]), cols[b]:(cols[b] + winDim[1])] + 1

            count += 1
    '''
    # Display the Heat map
    fig1 = plt.figure(1)
    plt.imshow(heatMap, cmap='bone')
    plt.title("Heat Map for the Image")
    plt.show()

    fig2 = plt.figure(2)
    plt.imshow(sum_map, cmap='bone')
    plt.title("Sum Map for the Image")
    plt.show()

    avg_map = np.divide(heatMap, sum_map)

    ax = plt.subplot()
    im = ax.imshow(avg_map)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    plt.show(block=False)
    '''
    return SNR_welch, SNR_ROIs


def aiq(image):
    # Create a window that is effectively a quarter of our image size.
    windowData = hanning_2d(image.shape, 0.25)  # makes the hanning window defined above
    '''
    # Step 1: Display the hann Window
    fig1 = plt.figure(1)
    plt.imshow(windowData, cmap='gray')
    plt.title("Hann Window")
    plt.show()
    '''

    welchPower, allTheROIs = welch_2d_windowing(image, windowData)  # Calling the function that creates the graph window
    # Need to make the below into a function so that this is done on each of the ROIs returned above - get an SNR for each ROI and then with those ROIs we will make a histogram with those

    SNR_image, SNR_List = calculateSNR(welchPower, allTheROIs, image, windowData)
    '''
    fig = plt.figure(5)
    plt.hist(SNR_List, range=[10, 45])
    plt.title("Histogram of all ROI")
    plt.show()

    file = open("HeatMapSNRs_FFR.txt", "w")
    for index in range(len(SNR_List)):
        file.write(str(SNR_List[index]) + "\n")

    file.close()
    
    thetasampling = 1
    rhosampling = 0.5

    # Our maximum valid radius is only as big as half our window size.
    maxrad = int(np.floor(welch_pwr_spect.shape[0] / 2)) + 1

    polar_spect = warp_polar(welch_pwr_spect, radius=maxrad, output_shape=(
        360 / thetasampling, maxrad / rhosampling))  # Change the coordinate space of the image
    # Remove the last column because it always seems to be all zeros anyway.
    polar_spect = polar_spect[0:180, :]

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

    # Step 6: Display the Average polar plot
    fig6 = plt.figure(6)
    plt.plot(freqBins, np.log10(polar_avg), color='c')
    plt.title("Polar Plot")

    signal_range = polar_avg[
        (spacing_bins <= low_noise_cutoff) & (spacing_bins >= high_noise_cutoff)]  # find the range that the noise is in
    total_range = polar_avg[(spacing_bins < high_noise_cutoff)]  # this is the total range of frequencies in the image

    # Use of the DERIVATIVE of the power spectrum captures any areas of interest that would cause sudden changes
    diffs = np.diff(polar_avg)

    signal_power_change = freq_bin_size * np.sum(abs(np.diff(signal_range)))
    noise_power_change = freq_bin_size * np.sum(abs(np.diff(total_range)))

    
    # Step 7: Take the derivative of the polar average
    fig7 = plt.figure(7)
    xs = np.linspace(1/low_noise_cutoff, 1/high_noise_cutoff, len(diffs) - 1)
    plt.plot(freqBins[1:], diffs, color='r', label='derivative')
    # plt.xlim([(1/low_noise_cutoff)-0.005, (1/high_noise_cutoff)+0.005])
    plt.ylim([-200, 200])
    plt.axvline(1/low_noise_cutoff, color='c', label='Low Cutoff')
    plt.axvline(1/high_noise_cutoff, color='y', label='High Cutoff')
    plt.title("Derivative of Polar Plot")

    plt.show(block=False)
    

    SNR = 10 * np.log10(signal_power_change / noise_power_change)
    '''
    return str(SNR_image)


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
    ovl = .25  # overlap between the ROIs 1- % you want
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
    # videos = get_files()    # used to get the video
    w = 0  # set to 1 in order to write the ROI
    SNRs = []
    sumR = 0
    q = 0  # set to 1 for ROI implementation and 2 for video implementation
    imNums = 1  # track the image numbers for naming purposes

    if q == 0:
        # fWhole = open("AOIP_Manuscript_Confocal_Images_SNR.txt", "w")
        fWhole = open("OCVL_Manuscript_Split_Images_SNR.txt", "w")
    elif q == 2:
        f_video = open("SNR_confocal_video_00-04710_vid_998.txt", "w")
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

        # Clear the entire list in case there are fewer ROIs generated in the next image and restart the sum
        SNRs.clear()
        sumR = 0

        # To write the ROI generated to their own tif files
        if w == 1:
            for t in range(R.shape[2]):
                name = "ROI" + str(t) + ".tif"
                savePath = "D:\Brea_Brennan\Image_Quality_Analysis\ROIs For Map\\" + name
                temp = R[..., t]
                if np.all(temp == 0):
                    print("All zeros! Do not right this ROI!")
                else:
                    cv2.imwrite(savePath, temp)

    # for v in videos:
    #     cap = cv2.VideoCapture(v)
    #     f_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     print(f_num)
    #     for f in range(0, f_num):
    #         ret, frame = cap.read()
    #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #
    #         # fig1 = plt.figure(1)
    #         # plt.imshow(frame, cmap='gray')
    #         # plt.title("Frame " + str(f))
    #         # plt.show()
    #
    #         SNR_frame = aiq(frame)
    #
    #         print("Frame image SNR: " + str(SNR_frame))
    #         f_video.write(str(SNR_frame) + "\n")

    if q == 0:
        fWhole.close()
    elif q == 2:
        f_video.close()
    else:
        fROI.close()

    print("End\n")
