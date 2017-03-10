from os import listdir
from params import Params
import pandas as pd
import fnmatch
import pickle
import numpy as np

__author__ = 'amanda'


def convert_ground_truth_to_df():
    p = Params(config_path='params.cfg')

    with open(p.cha_gt_raw_file, 'rb') as f:
        data = pickle.load(f)

    rating_array = np.zeros((6000, 6))
    for fea_ind, feat in enumerate(data):
        for sample_ind, file_name in enumerate(data[feat]):
            rating_array[sample_ind, fea_ind] = data[feat][file_name]

    cha_raw_gt = pd.DataFrame(data=rating_array, index=[i for i in data['extraversion']], columns=[j for j in data])
    cha_raw_gt.to_pickle(p.cha_gt_file)
    # How to load: df = pd.read_pickle(pickle_save_name)
    return


def make_sorted_gt():
    # only keep video snapshots that contain at least 10 face crops from each video.
    # the sorted gt will only include ground truth from videos that satistfies the minimal requirement.
    # im_list is a variable that saves a list of images, sorted by the ground truth's video sequence.
    # you may edit im_list to make a image list for caffe, or other frameworks.

    p = Params(config_path='params.cfg')
    cha_raw_gt = pd.read_pickle(p.cha_gt_file)
    new_gt = pd.DataFrame(columns=list(cha_raw_gt.columns.values))
    video_list = cha_raw_gt.index.tolist()

    face_img_names = listdir(p.crop_face_im_dir)
    im_list, im_count_list = [], []
    for cur_ind, cur_video in enumerate(video_list):
        if cur_ind % 100 == 0:
            print '{}-th video out of {}...'.format(cur_ind, len(video_list))
        pattern = cur_video[:-4]
        subset = fnmatch.filter(face_img_names, pattern+'*.jpg')
        if len(subset) > p.lower_bound_frame_num:
            im_list.extend(subset)
            im_count_list.append(len(subset))
            new_gt.loc[cha_raw_gt.index[cur_ind]] = cha_raw_gt.iloc[cur_ind]
        else:
            print 'video {} only contains {} image snapshots. Not included in the selection'.format(cur_video, len(subset))

    new_gt['img_frame_num'] = pd.Series(im_count_list, index=new_gt.index)
    new_gt.to_pickle(p.mapped_gt_file)
    print 'New gt length = {}'.format(len(new_gt))  # how many videos are kept.

    return


if __name__ == '__main__':
    convert_ground_truth_to_df()
    make_sorted_gt()
