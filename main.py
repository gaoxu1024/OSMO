# Final Version 1 -- with both auto-labeled obstacles and grount truth obstacles.

import torch
import numpy as np
from utils.easytest.utils_easytest20180307 import Net
from utils.functions import *
import scipy.io as sio
import argparse

def main():

    # Parameter Sets
    parser = argparse.ArgumentParser()
    parser.add_argument('--search_area_scale', type=int, default=30, help='The scale of obstacle searching area')
    parser.add_argument('--refine_thresh', type=float, default=0.4, help='The threshold of online track refinement')
    parser.add_argument('--is_refine_tracks', type=bool, default=True, help='Refine the tracks from tracking_methods or not')
    parser.add_argument('--bbox_tolerance', type=int, default=10, help='BBox size tolerance to measure two same BBox')
    parser.add_argument('--cost_thresh', type=float, default=3, help='If cost is lower than this param, object is still missing')
    parser.add_argument('--mean_vel_frames', type=float, default=5, help='Number of frames that have been used for motion estimation')
    parser.add_argument('--stop_if_missing_too_long', type=bool, default=True, help='If one object is missing too long, stop this trajectory')
    parser.add_argument('--hidden_layer_size', type=int, default=128, help='Size of RNN hidden layer')
    parser.add_argument('--use_ssm', '--verbose', help='Use Scene Structure Model or not', action='store_true')
    parser.add_argument('--method', type=str, default='CMOT', help='Method of initial tracking')
    parser.add_argument('--ds_no', type=int, default=3, help='Test Dataset number')
    args = parser.parse_args()

    print(args.use_ssm)

    # Basic
    ds = ['MetroOut', 'NightCrossing', 'CrowdedCrossing', 'CampusStone']
    ds_length = [800, 1000, 200, 1000]
    ds_w = [1920, 1920, 1920, 1560]
    ds_h = [1080, 1080, 1080, 1080]
    ds_no = args.ds_no
    method = args.method
    count = 0

    # Auto-labeled obstacles
    al_obst_addr = './al_obstacle/al_obstacle.mat'

    # Appnet initialization
    net = Net().cuda() if torch.cuda.is_available() else Net()
    model = torch.load('utils/model/model_finetune.pkl')
    net.load_state_dict(model.state_dict())

    obst = Obstacle()
    track_input_txt_addr = './track_input/' + method + '/' + ds[ds_no] + '.txt'
    det_txt_addr = './dataset/' + ds[ds_no] + '/0011/detection/det_result.txt'
    img_root_dir = './dataset/' + ds[ds_no] + '/0011/img/'
    init_track = np.genfromtxt(track_input_txt_addr, delimiter=',')
    all_det = np.genfromtxt(det_txt_addr, delimiter=',')
    online_track = []


    print('Generating obstacle map by auto-labeled obstacles ...................................')
    al_obst_temp = sio.loadmat(al_obst_addr)
    al_obst = al_obst_temp['al_obstacle']
    obst = get_al_obst_map(obst, al_obst, ds[ds_no], ds_w[ds_no], ds_h[ds_no], args)


    print('Washing data ...................................')
    init_track = wash_track_input(init_track, all_det, obst, args)

    # Online Refine MOT
    for f in range(0, ds_length[ds_no]):
        frame = f+1
        print('Processing frame % d ....................................' % frame)
        chosen_det = []  # Detections that have been chosen to associate one object in this frame.
        rest_det = []  # Detections that are about to associate to one object in this frame.

        online_track, chosen_det, rest_det = manage_online_track(online_track, init_track, chosen_det, rest_det, all_det, frame, img_root_dir, net, args, count, obst)  # Including appearance refinement

        if frame > 1:

            # Find those missing objects.
            online_track = manage_missing_object(online_track, frame)

            # Calculate the cost matrix between missing objects and the rest detections for data association (CORE PART)
            cost_matrix, missing_obj_idxes = cal_association_cost(online_track, rest_det, frame, net, obst, args, img_root_dir, ds_w[ds_no], ds_h[ds_no])

            # Use Hungarian algorithm for data association.
            association = []
            if not len(cost_matrix) == 0:
                association = hungarian(cost_matrix)
            print(association)

            # Update trajectories.
            online_track = update_online_track(online_track, rest_det, missing_obj_idxes, association, cost_matrix, frame, img_root_dir, net, args, obst)

    # Convert online_track into the txt file

    approach = 'AS' if args.use_ssm else 'A'
    save_dir = 'result/%s_%s_%s.txt' % (method, ds[ds_no], approach)
    save_online_track_into_txt(online_track, save_dir)
    smooth_track(save_dir)

if __name__ == '__main__':
    main()
