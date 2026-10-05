import AIQ
import cv2
import random

if __name__ == '__main__':
    videos = AIQ.get_files()
    filename = "random_10_conf_video_frames_snr_values.txt"
    video_names = "random_video_names.txt"
    f_video = open(filename, "w")
    f_vid_names = open(video_names, "w")

    for j in range(0, 10):
        rand_vid_num = random.randrange(0, len(videos))
        cap = cv2.VideoCapture(videos[rand_vid_num])
        f_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        f_vid_names.write(str(videos[rand_vid_num]) + "\n")

        for f in range(0, f_num):
            # open video and obtain frame in gray scale
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate the SNR value
            frame_snr = AIQ.aiq(frame)

            # write the SNR value to file
            f_video.write(str(frame_snr) + "\n")

            filename = videos[rand_vid_num].split("/")
            new_name = filename[-1][:-4] + "frame_" + str(f) + ".png"

            cv2.imwrite("P:\Brea_Brennan\Image_Quality_Analysis\Manuscript Data\Conf_video_all_frames\\" +
                            new_name, frame)

    f_video.close()
    f_vid_names.close()

    """
    for v in videos:
        cap = cv2.VideoCapture(v)
        f_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_bad = []
        frames_good = []

        for f in range(0, f_num):
            # open video and obtain frame in gray scale
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate the SNR value
            frame_snr = AIQ.aiq(frame)

            if float(frame_snr) < 25:
                frames_bad.append(frame)

            # write the SNR value to file
            f_video.write(str(frame_snr) + "\n")

        for i in range(0, len(frames_bad)):
            rand_num = random.randrange(0, len(frames_bad))
            rand_frame = frames_bad[rand_num]
            filename = v.split("/")
            new_name = filename[-1][:-4] + "frame_" + str(rand_num) + ".png"

            matches = [x for x in images if x == new_name]
            if len(matches) == 0:
                cv2.imwrite("P:\Brea_Brennan\Image_Quality_Analysis\Manuscript Data\confocal_bad_SNR\\" +
                            new_name, rand_frame)
        """