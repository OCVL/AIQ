import AIQ
import cv2

if __name__ == '__main__':
    videos = AIQ.get_files()
    filename = "video_frame_snr_values.txt"
    f_video = open(filename, "w")
    for v in videos:
        cap = cv2.VideoCapture(v)
        f_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        for f in range(0, f_num):
            # open video and obtain frame in gray scale
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate the SNR value
            frame_snr = AIQ.aiq(frame)

            # write the SNR value to file
            f_video.write(str(frame_snr) + "\n")