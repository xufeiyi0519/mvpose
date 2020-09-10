import os
import os.path as osp
import pickle
import sys
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter



project_root = os.path.abspath ( os.path.join ( os.path.dirname ( __file__ ), '..', '..' ) )
if __name__ == '__main__':
    if project_root not in sys.path:
        sys.path.append ( project_root )
import coloredlogs, logging

logger = logging.getLogger ( __name__ )
coloredlogs.install ( level='DEBUG', logger=logger )

from src.models.model_config import model_cfg
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from src.m_utils.base_dataset import BaseDataset, PreprocessedDataset
from src.models.estimate3d import MultiEstimator
from src.m_utils.evaluate import numpify
from src.m_utils.mem_dataset import MemDataset
from src.m_utils.visualize import plotPaper3d,plotPaper3dour, visualize, plotPaper3dold
from src.tools.getkey import get_key
from src.tools.withID_dist import withID_dist

def laplace_function(x, lambda_):
    return (1/(2*lambda_)) * np.e**(-1*(np.abs(x)/lambda_))
lam = 3
wei1 = laplace_function(0,lam)
wei2 = laplace_function(-1,lam)
wei3 = laplace_function(-2,lam)
wei_sum = wei1 + wei2 + wei3
weight1 = wei1/wei_sum
weight2 = wei2/wei_sum
weight3 = wei3/wei_sum

def tracking(posecur, pose, personid, img_id):

    posecurrent = posecur.copy()
    posepre = pose.copy()

    for i in range(len(posepre)):
        if len(posecurrent) == 0:
            break
        dist_ave = []
        for j in range(len(posecurrent)):
            dist = np.square(posecurrent[j] - posepre[i])
            dist = np.sqrt(dist[0] + dist[1] + dist[2])
            dist_ave.append(np.sum(dist) / 17)

        print(dist_ave)
        dist_min = np.min( np.array(dist_ave) )
        cur_id = dist_ave.index(dist_min)

        a = get_key(personid, posepre[i][0][0])
        tmp = personid[a]
        tmp.append(posecurrent[cur_id][0][0])
        personid[a] = tmp
        posecurrent.pop(cur_id)

    if len(posecurrent) != 0:
        for m in range(len(posecurrent)):
            personid[len(personid.keys())+m] = [posecurrent[m][0][0]]

    return personid


def export(model, loader, is_info_dicts=False, show=False):
    pose_list = list ()
    personid = dict()
    nums = 0
    count = 0
    change_frame = 0

    for img_id, imgs in enumerate ( tqdm ( loader ) ):
        try:
            pass
        except Exception as e:
            pass
            # poses3d = model.estimate3d ( img_id=img_id, show=False )
        if is_info_dicts:
            info_dicts = numpify ( imgs )

            model.dataset = MemDataset ( info_dict=info_dicts, camera_parameter=camera_parameter,
                                         template_name='Unified' )
            poses3d = model._estimate3d ( 0, show=show )
        else:
            this_imgs = list()
            for img_batch in imgs:
                this_imgs.append ( img_batch.squeeze ().numpy () )
                # print(this_imgs[0])
            poses3d = model.predict ( imgs=this_imgs, camera_parameter=camera_parameter, template_name='Unified',
                                          show=show, plt_id=img_id )
        # print(imgs[0])
        # print("pose")
        # print(poses3d)
        # print("111")
        pose_list.append ( poses3d )
        # print(pose_list)
        # print("111")
        # print(pose_list[-1])
        # print("1")





        nums = max(nums, len(poses3d))

#track
        if img_id == 0 :
            personid = { i:[poses3d[i][0][0]] for i in range(len(poses3d)) }
        else:

            if len(pose_list[img_id - 1]) == len(poses3d):
                tracking(poses3d,pose_list[img_id - 1], personid, img_id)
            elif len(pose_list[img_id - 1]) > len(poses3d):
                count += 1
                change_frame = img_id - count
                tracking(poses3d, pose_list[img_id - 2], personid, img_id)
            else:
                tracking(poses3d, pose_list[change_frame], personid, img_id)
                count = 0
        # laplace function
        if len(pose_list) >= 3:
            poses3D_1 = pose_list[-1]
            poses3D_2 = pose_list[-2]
            poses3D_3 = pose_list[-3]
            pose3D_1_dist = withID_dist(personid, poses3D_1)
            # print(pose3D_1_dist)
            pose3D_2_dist = withID_dist(personid, poses3D_2)
            # print(pose3D_2_dist)
            pose3D_3_dist = withID_dist(personid, poses3D_3)
            # print(pose3D_3_dist)
            # human_num1 = len(pose3D_1_dist)
            # human_num2 = len(pose3D_2_dist)
            # human_num3 = len(pose3D_3_dist)
            # human_num = min(human_num2,human_num3)

            pose_new = []
            # t = human_num1 - human_num
            # print(human_num1)
            # print(human_num2)
            # print(human_num3)
            # print(pose3D_1_dist)
            # if t <= 0:
            #
            #     for i in pose3D_1_dist.keys():
            #         pose_mid = pose3D_1_dist[i] * weight1 + pose3D_2_dist[i] * weight2 + pose3D_3_dist[i] * weight3
            #         pose_new.append(pose_mid)
            #         personid[i][-1] = pose_mid[0][0]
            # else:
            for i in pose3D_1_dist.keys():
                if i in pose3D_2_dist.keys() and i in pose3D_3_dist.keys():
                    pose_mid = pose3D_1_dist[i] * weight1 + pose3D_2_dist[i] * weight2 + pose3D_3_dist[i] * weight3
                    pose_new.append(pose_mid)
                    personid[i][-1] = pose_mid[0][0]
                else:
                    pose_mid = pose3D_1_dist[i]
                    pose_new.append(pose_mid)
                    personid[i][-1] = pose_mid[0][0]


            # print("new")
            # print(pose_new)
            # print("1")
            poses3d = pose_new
            pose_list.pop()
            pose_list.append(poses3d)
        # print("personid = ", personid)
        # print("poses3d = ",poses3d)
        print(personid)
        # print(poses3d)





#visualization
        fig_3d = plotPaper3dour(poses3d, personid)

        # fig_3d = plotPaper3dold(poses3d)

        # for n, cam in enumerate (model.dataset.cam_names):
        #     img = model.dataset.info_dict[cam]['image_data']
        #     img_init = visualize(img, return_img=True)
        #     # print(img_init)
        #     fig_3d.add_subplot(len(imgs), 2, 2 * cam + 1)
        #     plt.imshow(img_init)
        #     plt.xlabel(f'{cam}/{len(imgs)}')
        #     plt.xticks([])
        #     plt.yticks([])

        fig_3d.show()
        k = "%03d" % img_id
        fig_3d.savefig(f'/home/xfy/mvpose/3d/{k}.png')
# video
    path = '/home/xfy/mvpose/3d/'
    img_names = os.listdir(path)
    fps = 10
    size = (1280, 960)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video = cv2.VideoWriter('/home/xfy/mvpose/test1.avi', fourcc, fps, size)
    img_names.sort()

    for i in range(0, len(img_names)):
        print(img_names[i])
        img_path = os.path.join(path, img_names[i])
        img = cv2.imread(img_path)
        video.write(img)
    video.release()

    return pose_list


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser ()
    parser.add_argument ( '-d', nargs='+', dest='datasets', required=True,
                          choices=['Shelf', 'Campus', 'ultimatum1'] )
    parser.add_argument ( '-dumped', nargs='+', dest='dumped_dir', default=None )
    args = parser.parse_args ()

    test_model = MultiEstimator ( cfg=model_cfg )
    for dataset_idx, dataset_name in enumerate ( args.datasets ):
        model_cfg.testing_on = dataset_name
        if dataset_name == 'Shelf':
            dataset_path = model_cfg.shelf_path
            # you can change the test_rang to visualize different images (0~3199)
            test_range = range ( 0, 70, 1)
            gt_path = dataset_path

        elif dataset_name == 'Campus':
            dataset_path = model_cfg.campus_path
            # you can change the test_rang to visualize different images (0~1999)
            test_range = [i for i in range ( 105, 200, 1 )]
            gt_path = dataset_path

        else:
            logger.error ( f"Unknown datasets name: {dataset_name}" )
            exit ( -1 )

        # read the camera parameter of this dataset
        with open ( osp.join ( dataset_path, 'camera_parameter.pickle' ),
                    'rb' ) as f:
            camera_parameter = pickle.load ( f )

        # using preprocessed 2D poses or using CPN to predict 2D pose
        if args.dumped_dir:
            test_dataset = PreprocessedDataset ( args.dumped_dir[dataset_idx] )
            logger.info ( f"Using pre-processed datasets {args.dumped_dir[dataset_idx]} for quicker evaluation" )
        else:

            test_dataset = BaseDataset ( dataset_path, test_range )

        test_loader = DataLoader ( test_dataset, batch_size=1, pin_memory=True, num_workers=6, shuffle=False )
        pose_in_range = export ( test_model, test_loader, is_info_dicts=bool ( args.dumped_dir ), show=True )



        with open ( osp.join ( model_cfg.root_dir, 'result',
                               time.strftime ( str ( model_cfg.testing_on ) + "_%Y_%m_%d_%H_%M",
                                               time.localtime ( time.time () ) ) + '.pkl' ), 'wb' ) as f:
            pickle.dump ( pose_in_range, f )




