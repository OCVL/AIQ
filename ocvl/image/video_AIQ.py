import AIQ
import cv2
import numpy as np

if __name__ == '__main__':

    vids = AIQ.get_files()

    # f = open("video_SNR_investigation_all_confocal_vid_SNR_list.txt", "w")
    f = open("video_SNR_investigation_all_confocal_vid_SNR_list.csv", "w")
    # f2 = open("video_SNR_investigation_all_confocal_vid_SNR_frame_avg_list.txt", "w")
    average_SNR = []
    for v in vids:
        frame_snrs = []
        cap = cv2.VideoCapture(v)
        f_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f_num)
        parts = v.split('/')
        f.write("Video Name: " + parts[-1] + "\n")
        for b in range(0, f_num):
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            SNR_frame = AIQ.aiq(frame)
            frame_snrs.append(SNR_frame)

            print("Frame image SNR: " + str(SNR_frame))
            # f.write(str(SNR_frame) + "\n")
            f.write(str(SNR_frame) + ",")

        f.write("\n\n")
        # frame_snrs = [float(i) for i in frame_snrs]
        # avg_snr = np.sum(frame_snrs) / len(frame_snrs)
        # average_SNR.append(avg_snr)
        # f2.write(str(parts[-1]) + ", " + str(avg_snr) + "\n")

    f.close()
    # f2.close()
