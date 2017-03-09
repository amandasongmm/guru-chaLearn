import time
import cv2
from os.path import basename
import pandas as pd
from params import Params
import os
import numpy as np

__author__ = 'amanda'


def generate_mp4_list(p):
    f = open(p.mp4_list_file, 'w')
    for root, dirs, files in os.walk(p.mp4_root_dir):
        for one_file in files:
            if one_file.endswith('.mp4'):
                rel_file = os.path.join(root, one_file)
                f.write(rel_file + '\n')
    f.close()
    return


def sample_one_video(video_path, p, time_points, face_cascade):
    cur_video_name = basename(video_path)
    vid_cap = cv2.VideoCapture(video_path)
    full_im_num_count, face_im_num_count = 0, 0

    for frame_count, cur_tp in enumerate(time_points):
        vid_cap.set(cv2.cv.CV_CAP_PROP_POS_MSEC, cur_tp)
        success, image = vid_cap.read()
        if success:
            # save full image.
            if full_im_num_count < p.frames_per_video:
                full_im_file = '{}{}_frame{}.jpg'.format(p.crop_full_im_dir, cur_video_name[:-4], frame_count)
                full_im = image
                # full_im = cv2.resize(image, p.im_size)  # You may decide if to save resized image or the original one
                cv2.imwrite(full_im_file, full_im)
                full_im_num_count += 1

            # detect face from full image.
            if face_im_num_count < p.frames_per_video:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray,
                                                      scaleFactor=p.scale_factor,
                                                      minNeighbors=p.min_neighbors,
                                                      minSize=p.min_size,
                                                      flags=cv2.cv.CV_HAAR_SCALE_IMAGE)

                face_num = len(faces)  # number of faces detected in current image.
                if face_num == 0:
                    print 'No face detected in img {}'.format(cur_video_name[:-4])
                    break
                elif face_num == 1:
                    rect_ind = 0
                else:
                    rect_area = [i[2] * i[3] for i in faces]
                    rect_ind = np.argmax(rect_area)

                x, y, w, h = faces[rect_ind, :]
                crop_im = image[y:y + h, x:x + w]
                crop_im = cv2.resize(crop_im, p.im_size)
                face_im_file = '{}{}_frame{}.jpg'.format(p.crop_face_im_dir, cur_video_name[:-4], frame_count)
                cv2.imwrite(face_im_file, crop_im)
                face_im_num_count += 1

    return face_im_num_count


def sample_all_video():
    start_time = time.time()
    p = Params(config_path='params.cfg')
    generate_mp4_list(p)

    with open(p.mp4_list_file) as f:  # Iteratively process all 6000 videos.
        file_list = f.readlines()

    face_counts_per_video = []
    time_points = np.linspace(0, p.video_total_msec, p.upper_bound_frame_num)
    face_cascade = cv2.CascadeClassifier(p.face_detector_file)  # initialize face detector
    for i, cur_v_path in enumerate(file_list):
        v_path = cur_v_path.strip()
        face_count = sample_one_video(v_path, p, time_points, face_cascade)
        face_counts_per_video.append(face_count)
        if i % 20 == 0:
            print 'Current video num {} out of {}. Total elapsed time: {} seconds. \n'.\
                format(i, len(file_list), time.time() - start_time)

    df = pd.DataFrame({'filename': file_list, 'image_count': face_counts_per_video})
    df.to_pickle(p.sample_num_record_file)
    return


if __name__ == '__main__':
    sample_all_video()
