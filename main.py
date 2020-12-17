import matplotlib.pyplot as plt
import scipy.fft as scipy
import numpy as np
import cv2


def load_data():
    dimcap = cv2.VideoCapture(
        "M:\\Dropbox (Personal)\\Research\\Acquisition_Quality\\Data\\Dimming\\11002_20200107_confocal_OS_0026.avi")

    total_num_frms = dimcap.get(cv2.CAP_PROP_FRAME_COUNT)

    ret, frame = dimcap.read()

    print("Detected " + str(total_num_frms) + " frames that are: " + str(frame.shape[0]) + "x" + str(
        frame.shape[1]) + " in the video.")

    # Create our base array that'll hold the video.
    frame_dim = (frame.shape[0], frame.shape[1], int(total_num_frms))
    full_vid = np.empty(frame_dim, dtype=frame.dtype)

    i = 0
    while dimcap.isOpened():
        if not ret:
            break

        full_vid[:, :, i] = frame[:, :, 0] # These should be grayscale images only...

        ret, frame = dimcap.read()
        i += 1

    dimcap.release()
    cv2.destroyAllWindows()

    return full_vid


def rowwise_avg(vid):


    rowmeans = np.empty((vid.shape[0], vid.shape[2]))
    num_bins = 256
    histim = np.empty((num_bins, vid.shape[2]))

    for t in range(vid.shape[2]):
        rowmeans[:, t] = np.mean(vid[:, :, t], axis=1, dtype=np.float32)
        histim[:, t], edges = np.histogram(rowmeans[:, t], bins=num_bins, range=(0, 128))

    plt.imshow(histim, cmap="hot", clim=(0.0, 200))
    plt.title(str(t))
    plt.show()

def rowwise_fft(vid):


    rowmich = np.empty((vid.shape[0], vid.shape[2]))
    num_bins = 256
    histim = np.empty((num_bins, vid.shape[2]))

    T = 1.0/vid.shape[0]
    fft_samps = np.linspace(0.0, 1.0/(2.0*T), vid.shape[0]//2)

    for t in range(vid.shape[2]):
        for row in range(vid.shape[0]):
            rowfft = scipy.fft(vid[row, :, t])
            plt.plot(fft_samps, np.abs(rowfft[0:vid.shape[0]//2]))
            plt.draw()
        plt.show()





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    vidData = load_data()

    rowavgs = rowwise_avg(vidData)
    rowmich = rowwise_fft(vidData)




