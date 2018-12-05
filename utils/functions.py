import os
import math
import copy
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import easytest.utils_easytest20180307 as appnet  # Import parameters and network structure from the appearance RNN network.
from easytest.utils_easytest20180307 import Net
from munkres import Munkres
from utils import *
from PIL import Image, ImageDraw
import scipy.io as sio
import argparse
import matplotlib.pyplot as plt


def wash_track_input(init_track, all_det, obst, args):

    total_frame = max(init_track[:, 0])
    delete_list = []
    for f in range(int(total_frame)):
        for track_idx in np.argwhere(init_track[:, 0] == f):
            is_exist = False
            for det_idx in np.argwhere(all_det[:, 0] == f):
                track_idx = int(track_idx)
                det_idx = int(det_idx)

                judge = abs(init_track[track_idx, 2:6] - all_det[det_idx, 2:6]) <= args.bbox_tolerance

                if judge.all():
                    is_exist = True
                    break
            if not is_exist:  # If the bounding box is not in the original detection set
                for i in range(obst.count):

                    track_idx = int(track_idx)
                    obj_center = init_track[track_idx, 2:4] + 0.5 * init_track[track_idx, 4:6]
                    obst_bbox = obst.bbox[i]
                    obst_area = [obst_bbox[0], obst_bbox[1], obst_bbox[0] + obst_bbox[2], obst_bbox[1] + obst_bbox[3]]
                    if obst_area[0] < obj_center[0] < obst_area[2] and obst_area[1] < obj_center[1] < obst_area[3]:
                        delete_list.append(track_idx)
                        break
    delete_list.sort()

    for i in delete_list[::-1]:
        init_track = np.delete(init_track, i, 0)
    return init_track


def get_al_obst_map(obst, al_obst, ds_name, img_w, img_h, args):
    for i in range(len(al_obst)):

        if al_obst[i][0][0] == ds_name:
            obst.count = al_obst[i][1][0][0]
            obst.bbox = [[0 for idx in range(4)] for idx in range(obst.count)]
            obst.search_area = [[0 for idx in range(4)] for idx in range(obst.count)]
            obst.map = [[[0 for idx in range(img_w)] for idx in range(img_h)] for idx in range(obst.count)]
            obst_map = al_obst[i][3]

            for j in range(obst.count):
                x, y, w, h = al_obst[i][2][0][0][0][j][0:4]

                obst.bbox[j] = [x, y, w, h]
                scale_d = args.search_area_scale
                obst.search_area[j] = [x-scale_d, y-scale_d, w+scale_d*2, h+scale_d*2]
                obst_map_temp = [[0 for idx in range(img_w)] for idx in range(img_h)]
                obst_map_temp = np.array(obst_map_temp)
                obst_map_temp[y:(y+h), x:(x+w)] = obst_map[y:(y+h), x:(x+w)]
                obst.map[j] = obst_map_temp.tolist()

    return obst


def cal_triangle_area(pa, pb, pc):
    v1 = pb - pa
    v2 = pc - pa
    v1_mo = np.sqrt(v1.dot(v1))
    v2_mo = np.sqrt(v2.dot(v2))
    cos_theta = np.dot(v1, v2) / (v1_mo * v2_mo)
    S = 0.5 * v1_mo * v2_mo * np.sqrt(1 - pow(cos_theta, 2))
    return S


def cal_association_cost(online_track, rest_det, frame, net, obst, args, img_root_dir, img_width, img_height):
    missing_obj_idxes = []
    general_prob_ob = -1
    for num, track in enumerate(online_track):
        if track.is_missing:
            missing_obj_idxes.append(num)

    cost_matrix = [[-1 for _ in range(len(rest_det))] for _ in range(len(missing_obj_idxes))]  # Initialization.

    for i, mobj_idx in enumerate(missing_obj_idxes):

        # For each missing object in the searching area of the obstacle, calculate the prediction of the missing object.
        if args.use_ssm:
            if len(online_track[mobj_idx].missing_pred) == 0:
                online_track[mobj_idx] = cal_missing_prediction_in_ssm(online_track[mobj_idx], obst, general_prob_ob, img_width, img_height, args)

        for j, rdet in enumerate(rest_det):

            seq_app = extract_track_app_feature(online_track[mobj_idx], rdet.fmap, frame, img_root_dir, net, args)
            prob_app = cal_appearance_similarity(seq_app, rdet.app, net)
            prob_motion = cal_motion_similarity(online_track[mobj_idx], rdet.pos, frame, args)

            if args.use_ssm:

                prob_ob = cal_scene_structure_similarity(online_track[mobj_idx], rdet, frame, general_prob_ob, args)

                if prob_ob == general_prob_ob:
                    cost_matrix[i][j] = - math.log(prob_app) - math.log(prob_motion)
                else:
                    cost_matrix[i][j] = - math.log(prob_app) - math.log(prob_ob)
            else:
                cost_matrix[i][j] = - math.log(prob_app) - math.log(prob_motion)

    return cost_matrix, missing_obj_idxes


def cal_missing_prediction_in_ssm(obj, obst_all, general_prob_ob, img_width, img_height, args):

    w = img_width
    h = img_height

    if obj.is_missing:  # If the object is a missing object

        if len(obj.missing_pred) == 0:  # Build the prediction for those missing object when occluded by obstacles.
            missing_frame = obj.frame[-1]

            # Obstacle Constraint
            # Find the obstacle searching area the object belongs to.
            obj_pos = obj.pos[-1]
            obj_center = [int(obj_pos[0] + obj_pos[2] * 0.5), int(obj_pos[1] + obj_pos[3] * 0.5)]
            iou_max, obst_idx_of_obj = -1, -1

            for obst_idx in range(obst_all.count):
                search_area = obst_all.search_area[obst_idx]
                obst_area = obst_all.bbox[obst_idx]

                # If the missing object is located in the searching area and not on the obstacle (The latter one ignores those object that is in front of the obstacles.)
                if search_area[0] < obj_center[0] < search_area[0] + search_area[2] and search_area[1] < obj_center[1] < search_area[1] + search_area[3]:
                    if not (obst_area[0] < obj_center[0] < obst_area[0] + obst_area[2] and obst_area[1] < obj_center[1] < obst_area[1] + obst_area[3]):
                        iou = cal_iou(search_area, obj_pos)
                        if iou > iou_max:
                            obst_idx_of_obj = obst_idx
                            iou_max = iou

            if obst_idx_of_obj == -1 or (obj.vel[0] == 0 and obj.vel[1] == 0):  # If the missing object is not in all of the obstacle searching areas or vel=0
                prob_ob = general_prob_ob
            else:  # If the missing object is in the obstacle searching areas
                map = np.array(obst_all.map[obst_idx_of_obj])

                obj_anchor_set = []
                obj_center = [int(obj_pos[0] + obj_pos[2] * 0.5), int(obj_pos[1] + obj_pos[3] * 0.5)]
                obj_ul = [int(obj_pos[0]), int(obj_pos[1])]
                obj_ur = [int(obj_pos[0] + obj_pos[2]), int(obj_pos[1])]
                obj_dl = [int(obj_pos[0]), int(obj_pos[1] + obj_pos[3])]
                obj_dr = [int(obj_pos[0] + obj_pos[2]), int(obj_pos[1] + obj_pos[3])]
                obj_anchor_set.append(obj_center)
                obj_anchor_set.append(obj_ul)
                obj_anchor_set.append(obj_ur)
                obj_anchor_set.append(obj_dl)
                obj_anchor_set.append(obj_dr)

                obj_anchor_idx = 0
                for num, obj_anchor in enumerate(obj_anchor_set):
                    obst_label_last = 0
                    record_pos = []
                    for posx in range(map.shape[1]):
                        if obj.vel[0] == 0:
                            posy = 1e-5
                        else:
                            posy = obj_anchor[1] + (float(obj.vel[1]) / float(obj.vel[0])) * (posx - obj_anchor[0])

                        if int(posy) < 0 or int(posy) >= map.shape[0]:
                            continue

                        obst_label = map[int(posy), int(posx)]

                        if 0 <= posx < map.shape[1] and 0 <= posy < map.shape[0]:
                            if obst_label_last == obst_label:
                                continue
                            else:
                                record_pos.append([posx, posy])

                        obst_label_last = obst_label

                    if len(record_pos) != 0:
                        obj_anchor_idx = num
                        break

                if len(record_pos) != 0:
                    obst_anchor = [0, 0]
                    obj_anchor = obj_anchor_set[obj_anchor_idx]
                    obst_anchor[0] = 0.5 * (record_pos[0][0] + record_pos[-1][0])
                    obst_anchor[1] = 0.5 * (record_pos[0][1] + record_pos[-1][1])

                    relative_position = np.array(obst_anchor) - np.array(obj_anchor)

                    if (relative_position[0] * obj.vel[0] + relative_position[1] * obj.vel[1]) <= 0:  # Object is not towards to the obstacle.
                        prob_ob = general_prob_ob
                    else:
                        pred_pos = copy.deepcopy(obj_pos)
                        pred_frame = missing_frame
                        iter = 0
                        while 1:

                            pred_pos[0] += obj.vel[0]
                            pred_pos[1] += obj.vel[1]
                            pred_frame += 1
                            pred_center = [int(pred_pos[0] + pred_pos[2] * 0.5), int(pred_pos[1] + pred_pos[3] * 0.5)]

                            pred_map = map[max(int(pred_pos[1]), 0): min(int(pred_pos[1] + pred_pos[3]), h), max(int(pred_pos[0]), 0): min(int(pred_pos[0] + pred_pos[2]), w)]

                            vector1 = np.array(obst_anchor) - np.array(obj_center)
                            vector2 = np.array(pred_center) - np.array(obst_anchor)

                            iter += 1

                            judge = float(np.sum(pred_map)) / max(float(pred_map.shape[0] * pred_map.shape[1]), 0.1)

                            if (vector1[0] * vector2[0] + vector1[1] * vector2[1]) > 0 and judge < 0.3:
                                break
                            if iter > 300:
                                break
                        missing_prediction = [0, 0, 0, 0, 0, 0]
                        missing_prediction[0] = pred_frame
                        missing_prediction[1:5] = pred_pos
                        missing_prediction[5] = obst_idx_of_obj
                        obj.missing_pred = missing_prediction
    return obj


def cal_appearance_similarity(obj_app, det_app, net):
    prob_app = net.metric_fc(torch.abs(obj_app - det_app))
    return float(prob_app)


def cal_motion_similarity(obj, det_pos, cur_frame, args):

    obj_pos = obj.pos
    obj_frame = obj.frame
    vel = obj.vel

    pred_pos = copy.deepcopy(obj_pos[-1])
    pred_center = np.array([pred_pos[0]+0.5*pred_pos[2], pred_pos[1]+0.5*pred_pos[3]])
    det_center = np.array([det_pos[0]+0.5*det_pos[2], det_pos[1]+0.5*det_pos[3]])
    if len(obj_pos) > 1:
        pred_center += np.array(vel) * min(abs(cur_frame - obj_frame[-1]), 5)
    prob_motion = math.exp(-pow(np.linalg.norm(pred_center - det_center), 2)/3000)

    return max(prob_motion, 1e-100)


def cal_scene_structure_similarity(obj, det, cur_frame, general_prob_ob, args):
    prob_ob = general_prob_ob

    if obj.is_missing:  # If the object is a missing object

        if len(obj.missing_pred) != 0:

            pred_frame = obj.missing_pred[0]
            pred_pos = obj.missing_pred[1:5]
            pred_center = np.array([int(pred_pos[0] + pred_pos[2] * 0.5), int(pred_pos[1] + pred_pos[3] * 0.5)])
            det_center = np.array([int(det.pos[0] + det.pos[2] * 0.5), int(det.pos[1] + det.pos[3] * 0.5)])

            prob_ob_spacial = math.exp(-pow(np.linalg.norm(pred_center - det_center), 2) / 3000)
            prob_ob_temporal = math.exp(-pow(cur_frame - pred_frame, 2) / 10000)
            prob_ob = prob_ob_spacial * prob_ob_temporal
            prob_ob = max(prob_ob, 1e-100)

    return prob_ob


def extract_target_app_feature(bbox, frame, img_root_dir, net):
    Transform = transforms.Compose([transforms.ToTensor()])
    img_dir = img_root_dir + '%0.6d.jpg' % frame
    img = Image.open(img_dir).convert('RGB')
    region = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
    crop_img = img.crop(region)
    crop_img = crop_img.resize((appnet.img_h, appnet.img_w))
    input_data = Transform(crop_img)
    input_data = input_data.view(1, input_data.shape[0], input_data.shape[1], input_data.shape[2],)

    if torch.cuda.is_available():
        input_data = input_data.cuda()

    bbox_fmap = net.feature_map(Variable(input_data, volatile=True))
    bbox_app = net.feature_fc(net.feature_pool(bbox_fmap).view(bbox_fmap.shape[0], -1))
    bbox_app = net.fc(bbox_app)

    return bbox_app, bbox_fmap


def extract_track_app_feature(seq, target_fmap, frame, img_root_dir, net, args):

    # Add the last bbox from this track into the appearance feature (RNNCell).
    Transform = transforms.Compose([transforms.ToTensor()])
    img_dir = img_root_dir + '%0.6d.jpg' % seq.frame[-1]
    img = Image.open(img_dir).convert('RGB')

    seq_len = min(len(seq.frame), 20)
    input_data = torch.FloatTensor(1, seq_len, 3, appnet.img_h, appnet.img_w)
    for i, num in enumerate(range(len(seq.frame) - seq_len, len(seq.frame))):
        bbox = seq.pos[num]
        region = (bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
        crop_img = img.crop(region)
        crop_img = crop_img.resize((appnet.img_h, appnet.img_w))
        input_img = Transform(crop_img)

        input_data[0][i] = input_img

    if torch.cuda.is_available():
        input_data = input_data.cuda()
    input_data = Variable(input_data, volatile=True)

    h = Variable(torch.zeros(1, args.hidden_layer_size)).cuda()  # Initialize the hidden layer h
    c = Variable(torch.zeros(1, args.hidden_layer_size)).cuda()  # Initialize the hidden layer c

    for frame in range(seq_len):

        xt_fmap = net.feature_map(input_data[:, frame])
        xt_input = net.feature_fc(net.feature_pool(xt_fmap).view(xt_fmap.shape[0], -1))

        diff_map = torch.abs(xt_fmap - target_fmap)
        diff_map = net.metric_subnet_conv(diff_map)
        diff_map = diff_map.view(diff_map.shape[0], -1)
        prob = net.metric_subnet_fc(diff_map)
        current_input = prob * xt_input + (1 - prob) * h  # Balance the current frame and previous frames
        h, c = net.rnncell(current_input, (h, c))

    x_app = h  # Output of the RNN network

    return x_app


def manage_online_track(online_track, init_track, chosen_det, rest_det, all_det, frame, img_root_dir, net, args, count, obst):  # Including appearance refinement
    for idx in np.argwhere(init_track[:, 0] == frame):
        idx = int(idx)
        button = -1

        for j in range(len(online_track)):
            if int(init_track[idx][1]) == online_track[j].id:
                button = j
                break

        if button == -1 and frame == 1:
            # Create a new track
            online_track_temp = Track()
            online_track_temp.pos = []
            online_track_temp.frame = []
            online_track_temp.pos.append(init_track[idx][2:6])
            online_track_temp.frame.append(init_track[idx][0])
            online_track_temp.id = int(init_track[idx][1])
            online_track_temp.vel = cal_track_velocity(online_track_temp, args)
            online_track.append(online_track_temp)

        if button == -1 and frame != 1:
            # Take this track as the rest detection for association.
            daiding_det_temp = Detection()
            daiding_det_temp.pos = init_track[idx][2:6]
            daiding_det_temp.app, daiding_det_temp.fmap = extract_target_app_feature(init_track[idx][2:6], frame, img_root_dir, net)
            daiding_det_temp.is_associated = False
            daiding_det_temp.id = init_track[idx][1]
            rest_det.append(daiding_det_temp)

        if button != -1:

            online_track[button].vel = cal_track_velocity(online_track[button], args)

            app_similarity, motion_similarity, count = refine_online_track_by_app(online_track[button], init_track[idx], img_root_dir, frame, count, net, args)

            if args.is_refine_tracks:
                if app_similarity >= args.refine_thresh and motion_similarity >= args.refine_thresh/8:
                    # Add this position onto that track
                    online_track[button].pos.append(init_track[idx][2:6])
                    online_track[button].frame.append(init_track[idx][0])
                    online_track[button].is_missing = False
                    online_track[button].missing_pred = []
                    chosen_det.append(init_track[idx][2:6])
                else:
                    # Add this position into the daiding_det
                    daiding_det_temp = Detection()
                    daiding_det_temp.pos = init_track[idx][2:6]
                    daiding_det_temp.app, daiding_det_temp.fmap = extract_target_app_feature(init_track[idx][2:6], frame, img_root_dir, net)
                    daiding_det_temp.is_associated = False
                    daiding_det_temp.id = init_track[idx][1]
                    rest_det.append(daiding_det_temp)

            else:
                online_track[button].pos.append(init_track[idx][2:6])
                online_track[button].frame.append(init_track[idx][0])
                online_track[button].is_missing = False
                online_track[button].missing_pred = []
                chosen_det.append(init_track[idx][2:6])

    return online_track, chosen_det, rest_det


def manage_missing_object(online_track, frame):  # frame > 1
    for i in range(len(online_track)):
        if frame-1 in online_track[i].frame and frame not in online_track[i].frame:
            online_track[i].is_missing = True
    return online_track


def refine_online_track_by_app(seq, target, img_root_dir, frame, count, net, args):  # Split wrong associations in the online tracks by appearance.

    target_app, target_fmap = extract_target_app_feature(target[2:6], target[0], img_root_dir, net)
    seq_app = extract_track_app_feature(seq, target_fmap, frame, img_root_dir, net, args)
    app_similarity = cal_appearance_similarity(seq_app, target_app, net)
    motion_similarity = cal_motion_similarity(seq, target[2:6], frame, args)

    return float(app_similarity), float(motion_similarity), count


def update_online_track(online_track, rest_det, missing_obj_idxes, association, cost_matrix, frame, img_root_dir, net, args, obst):

    if len(association) != 0:
        for asso in association:
            if cost_matrix[asso[0]][asso[1]] < args.cost_thresh:
                print cost_matrix[asso[0]][asso[1]]
                if frame == online_track[missing_obj_idxes[asso[0]]].frame[-1]:
                    online_track[missing_obj_idxes[asso[0]]].pos[-1] = rest_det[asso[1]].pos
                    online_track[missing_obj_idxes[asso[0]]].is_missing = False
                    online_track[missing_obj_idxes[asso[0]]].missing_pred = []
                    if rest_det[asso[1]].id != -1:
                        online_track[missing_obj_idxes[asso[0]]].id = rest_det[asso[1]].id
                    rest_det[asso[1]].is_associated = True
                else:
                    online_track[missing_obj_idxes[asso[0]]].pos.append(rest_det[asso[1]].pos)
                    online_track[missing_obj_idxes[asso[0]]].frame.append(frame)
                    online_track[missing_obj_idxes[asso[0]]].is_missing = False
                    online_track[missing_obj_idxes[asso[0]]].missing_pred = []
                    if rest_det[asso[1]].id != -1:
                        online_track[missing_obj_idxes[asso[0]]].id = rest_det[asso[1]].id
                    rest_det[asso[1]].is_associated = True

    # Add detections that are not associated to any object as new object.
    for rdet in rest_det:
        if not rdet.is_associated:
            online_track_temp = Track()
            online_track_temp.pos = []
            online_track_temp.frame = []
            online_track_temp.pos.append(rdet.pos)
            online_track_temp.frame.append(frame)

            tid = -999
            for i in range(len(online_track)):
                if online_track[i].id == rdet.id:
                    tid = i

            if tid != -999:
                online_track[tid].id = len(online_track) + 999999

            online_track_temp.id = rdet.id  # Special mark. (-1 or new ID)
            online_track_temp.vel = cal_track_velocity(online_track_temp, args)
            online_track.append(online_track_temp)

    # Shut down those missing online track that has been missing over a period of time.
    if args.stop_if_missing_too_long:
        for track in online_track:
            # If use ssm constraint and the track is missing from the obstacle, set a longer searching period.
            if args.use_ssm and hasattr(track, 'missing_pred'):
                if abs(frame - track.frame[-1]) >= 60:
                    track.is_missing = False
                    track.missing_pred = []
            else:
                if abs(frame - track.frame[-1]) >= 15:
                    track.is_missing = False

    return online_track


def save_online_track_into_txt(online_track, save_dir):
    with open(save_dir, 'w') as f:
        for num, traj in enumerate(online_track):
            track_id = num + 1
            for i, pos in enumerate(traj.pos):
                frame = traj.frame[i]
                output_text = '%d,%d,%d,%d,%d,%d,-1,-1,-1,-1\n' % (frame, track_id, int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3]))
                f.write(output_text)


def smooth_track(save_dir):
    with open(save_dir, 'a') as f:
        track = np.genfromtxt(save_dir, delimiter=',')
        tracking_id = int(max(track[:, 1]))
        for id in range(tracking_id):
            idx = np.argwhere(track[:, 1] == id)
            for i in range(len(idx)):
                if i != 0:
                    gap = abs(track[idx[i], 0] - track[idx[i - 1], 0])
                    if gap != 1 and gap <= 15:
                        change = (track[idx[i], 2:6] - track[idx[i-1], 2:6])/gap
                        for g in range(gap-1):
                            frame = track[idx[i - 1], 0] + (g+1)
                            new_track_pos = track[idx[i-1], 2:6] + (g+1) * change
                            output_text = '%d,%d,%d,%d,%d,%d,-1,-1,-1,-1\n' % (frame, id, int(new_track_pos[0][0]), int(new_track_pos[0][1]), int(new_track_pos[0][2]), int(new_track_pos[0][3]))
                            f.write(output_text)

