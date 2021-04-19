import torch
import torchvision
import os
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from plotting import plot_frame, plot_frame_one_row, get_roi_pts
from utils import ROIs, find_violation
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

from utils import COCO_INSTANCE_CATEGORY_NAMES as LABELS
import cv2
import glob
from siamfc import TrackerSiamFC

np.set_printoptions(precision=4)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

detector = 'faster_rcnn'


def main(dataset, data_time, detector):

    path_result = os.path.join('results', data_time + '_' + detector, dataset)
    os.makedirs(path_result, exist_ok=True)

    # initialize detector
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.to(device=device)
    model.eval()

    # load background
    img_bkgd_bev = cv2.imread('calibration/' + dataset + '_background_calibrated.png')
    # load transformation matrix
    transform_cam2world = np.loadtxt('calibration/' + dataset + '_matrix_cam2world.txt')

    # open video of dataset
    if dataset == 'oxford_town':
        cap = cv2.VideoCapture(os.path.join('datasets', 'TownCentreXVID.avi'))
        frame_skip = 10  # oxford town dataset has fps of 25
        thr_score = 0.9
    elif dataset == 'oxford_town_group':
        cap = cv2.VideoCapture(os.path.join('datasets', 'TownCentreXVID.avi'))
        path_track_frames = os.path.join(os.getcwd(),'datasets', 'dataset_tracks', 'TownCentre', 'img1')
        frame_images = sorted(glob.glob(path_track_frames + '\****.jpg'))
        net_path = os.path.join(os.getcwd(),'tracker','siamfc_pytorch','tools','pretrained\siamfc_alexnet_e50.pth')
        tracker = TrackerSiamFC(net_path=net_path)
        frame_skip = 10  # oxford town dataset has fps of 25
        thr_score = 0.9
    elif dataset == 'mall':
        cap = cv2.VideoCapture(os.path.join('datasets', 'mall.mp4'))
        frame_skip = 1
        thr_score = 0.9
    elif dataset == 'grand_central':
        cap = cv2.VideoCapture(os.path.join('datasets', 'grandcentral.avi'))
        frame_skip = 25  # grand central dataset has fps of 25
        thr_score = 0.5
    else:
        raise Exception('Invalid Dataset')

    # f = open(os.path.join(path_result, 'statistics.txt'), 'w')



    statistic_data = []
    i_frame = 0
    # while cap.isOpened() and i_frame < 5000:
    while cap.isOpened() and i_frame <=  7450:
        ret, img = cap.read()
        print("at frame " + str(i_frame) + "------")
        if ret is False:
            break
        if i_frame % frame_skip == 0: #only run the social distancing system every 10 frames.
            #ret, img = cap.read()
            # print('Frame %d - ' % i_frame)
            # if i_frame > 50:
            #     break

            # skip frames to achieve 1hz detection
            # if not i_frame % frame_skip == 0:  # conduct detection per second
            #     i_frame += 1
            #     continue

            #vis = True
            if i_frame <= 3000:
            # if i_frame / frame_skip < 20:
                vis = True
            else:
                vis = False

            # counting process time
            t0 = time.time()

            # convert image from OpenCV format to PyTorch tensor format
            img_t = np.moveaxis(img, -1, 0) / 255
            img_t = torch.tensor(img_t, device=device).float()

            # pedestrian detection
            predictions = model([img_t])
            boxes = predictions[0]['boxes'].cpu().data.numpy()
            classIDs = predictions[0]['labels'].cpu().data.numpy()
            scores = predictions[0]['scores'].cpu().data.numpy()
            box_id = [0]*len(boxes); # array to hold box ids for tracking

            #box 1 at (x1,y1), (x2,y2)
            #box 2 at (x1,y1), (x2,y2)
            #reg box_1 array - box_2 array

            # get positions and plot on raw image
            pts_world = []
            iter_tracks = []
            for i in range(len(boxes)):
                ##if class is a person and threshold is met
                if classIDs[i] == 1 and scores[i] > thr_score:
                    # extract the bounding box coordinates
                    (x1, y1) = (boxes[i][0], boxes[i][1])
                    (x2, y2) = (boxes[i][2], boxes[i][3])

                    #detector gives coords x1 ,y1, x2, y2
                    #convert these coords to tracker input
                    #input for tracker is a bounding box [x1,y1, width, height]
                    track_box_in = [boxes[i][0], boxes[i][1], boxes[i][2]-boxes[i][0], boxes[i][3]-boxes[i][1]]

                    #adjust so input images are 3 frames at i, i +5, i +10 rather than the whole set
                    track_images = []
                    for z in range(4): #number of frames to prepare
                        if i+5*z < len(frame_images):
                            track_images.append(frame_images[i_frame+5*z])


                    ##step 1 tracker
                    curr_track = tracker.track(track_images, track_box_in)
                    ##assign labels to the bounding boxes
                    ##label box = x
                    box_id[i] = i+1;
                    iter_tracks.append(curr_track)
                    #run tracker on each box

                    ################################################
                    #(if box_id > 0 run tracker)
                        #takes box (pixel coord)
                        #10 frames skip
                        #output is a tracklet area of the coord in each frame
                    #if ( box_id[i] > 0):




                    #convert coord of tracket to real

                    #run regression on coord of traklet if their is a violation

                    #regression confidence is high that difference is low that means no violation
                    #regress(y1,y2 -> x1,x2 from the difference array


                    ##############################################################################

                    if vis:
                        # draw a bounding box rectangle and label on the image
                        cv2.rectangle(img, (x1, y1), (x2, y2), [0, 0, 255], 2)

                        text = "{}: {:.2f}".format(LABELS[classIDs[i]], scores[i], box_id[i])
                        cv2.putText(img, text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 2)

                    # find the bottom center position and convert it to world coordinate
                    p_c = np.array([[(x1 + x2)/2], [y2], [1]])
                    p_w = transform_cam2world @ p_c
                    p_w = p_w / p_w[2]
                    pts_world.append([p_w[0][0], p_w[1][0]])

                ## convert all tracks coords to real world
            track_btm_cntr = np.zeros((len(iter_tracks),4,3)) ##to hold bounding boxes adjusted to bottom center coord
            track_world = np.zeros((len(iter_tracks),4,3)) ##to hold real world coord
            ## for each track iterate through each bounding box in a track and convert it to realworld coord
            ## foll steps above in which p_c and p_w are calculated
            for w in range(len(iter_tracks)):
                for u in range(4): #add each of the boxes from the 4 frame of the track
                    row_converted = np.array(
                        [[(iter_tracks[w][0][u][0] + iter_tracks[w][0][u][0] + iter_tracks[w][0][u][2]) / 2],
                        [iter_tracks[w][0][u][1] + iter_tracks[w][0][u][3]], [1]])
                    track_btm_cntr[w][u] = [row_converted[0], row_converted[1], row_converted[2]]
                    track_world[w][u] = transform_cam2world @ track_btm_cntr[w][u]
                    track_world[w][u] = track_world[w][u] / track_world[w][u][2]

            #get every combination of difference between each track
            #because difference between track i and track j is the just the negative of the difference of track j and i
            #only store i - j
            track_differences = { (w,u):0 for w in range(len(track_world)-1)  for u in range(1+w,len(track_world)) }
            for w in range(len(track_differences)):
                for u in range(w+1,len(track_world)):
                    track_diff_w = track_world[w,:,:2]
                    track_diff_u = track_world[u,:,:2]
                    track_diff = track_diff_w - track_diff_u
                    track_differences[w,u]= track_diff

            ##regress each item in the difference dictionary against 0. If the p > 0.05 we fail to reject that their
            ##there is a difference between two tracks (that is to say they are walking together)

            #holds the outcome for a track pair (i,j) if they are a group or not
            track_regression_out = { (w,u):0 for w in range(len(track_world)-1)  for u in range(1+w,len(track_world)) }
            for pair in track_differences:
                pair_x = track_differences[pair][:,0].reshape(-1,1)
                pair_y = track_differences[pair][:, 1].reshape(-1,1)
                pair_norm = [0,0,0,0]
                x_sample = [1,2,3,4]
                for j in range(4):
                    if j == 0:
                        pair_norm[j] = np.linalg.norm([pair_x[j],pair_y[j]])*0.001
                    else:
                        pair_norm[j] = np.linalg.norm([pair_x[j],pair_y[j]]) - np.linalg.norm([pair_x[0],pair_y[0]])
                reg_pair = sm.OLS(pair_norm, x_sample)
                #reg_pair = sm.OLS(pair_y, pair_x)
                reg_pair = reg_pair.fit()
                p_value = reg_pair.pvalues
                if pair == (5,6):
                    x = "test"
                #if pvalue is less than 0.05 we reject null in favour that is there is a difference between track i and j
                #so they are not a group and set track_regression_out to false
                if p_value < 0.05:
                    track_regression_out[pair] = False
                #else set track regression out to true because we fail to reject the null and therefore conclude that
                #the two tracks are a group
                else:
                    track_regression_out[pair] = True

            t1 = time.time()

            pts_world = np.array(pts_world)
            if dataset == 'oxford_town':
                pts_world[:, [0, 1]] = pts_world[:, [1, 0]]
                pass
            elif dataset == 'oxford_town_group':
                pts_world[:, [0, 1]] = pts_world[:, [1, 0]]
                pass
            elif dataset == 'mall':
                # pts_world[:, [0, 1]] = pts_world[:, [1, 0]]
                pass
            elif dataset == 'grand_central':
                # pts_world[:, [0, 1]] = pts_world[:, [1, 0]]
                pass


            statistic_data.append((i_frame, t1 - t0, pts_world, track_regression_out))

            # visualize
            if vis:
                violation_pairs = find_violation(pts_world, track_regression_out)
                pts_roi_world, pts_roi_cam = get_roi_pts(dataset=dataset, roi_raw=ROIs[dataset], matrix_c2w=transform_cam2world)

                fig = plot_frame_one_row(
                    dataset=dataset,
                    img_raw=img,
                    pts_roi_cam=pts_roi_cam,
                    pts_roi_world=pts_roi_world,
                    pts_w=pts_world,
                    pairs=violation_pairs
                )

                # fig = plot_frame(
                #     dataset=dataset,
                #     img_raw=img,
                #     img_bev_bkgd_10x=img_bkgd_bev,
                #     pts_roi_cam=pts_roi_cam,
                #     pts_roi_world=pts_roi_world,
                #     pts_w=pts_world,
                #     pairs=violation_pairs
                # )

                fig.savefig(os.path.join(path_result, 'frame%04d.png' % i_frame))
                plt.close(fig)

            # update loop info
            print('Frame %d - Inference Time: %.2f' % (i_frame, t1 - t0))
            print('=======================')
        i_frame += 1


    if cap.isOpened():
        cap.release()
    # save statistics
    # f.close()
    pickle.dump(statistic_data, open(os.path.join(path_result, 'statistic_data.p'), 'wb'))


if __name__ == '__main__':
    data_time = 'test'
    # data_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    #datasets = ['oxford_town', 'grand_central', 'mall']
    #datasets = ['oxford_town']
    datasets = ['oxford_town_group']
    #datasets = ['mall']

    for dataset in datasets:
        print('=========== %s ===========' % dataset)
        main(dataset=dataset, data_time=data_time, detector=detector)