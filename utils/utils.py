from munkres import Munkres


def cal_track_velocity(obj, args):

    do_update = True

    if len(obj.pos) == 1:
        vel_mean = [0, 0]
    else:

        if do_update:

            vel_sum = [0, 0]
            last_frame = -1
            count = -1

            for i in range(len(obj.frame))[::-1]:
                if obj.frame[i] == obj.frame[-1]:
                    last_frame = obj.frame[i]
                    continue

                f1, f2 = int(obj.frame[i]), int(obj.frame[i+1])  # f1 < f2
                fidx1, fidx2 = i, i+1

                vel = (obj.pos[fidx2][0:2]+0.5*obj.pos[fidx2][2:4]) - (obj.pos[fidx1][0:2]+0.5*obj.pos[fidx1][2:4])
                vel_sum += vel
                count = abs(last_frame - f1)

                if abs(last_frame - f1) > args.mean_vel_frames:
                    break
            vel_mean = vel_sum/count

        else:
            vel_mean = obj.vel

    return vel_mean


def cal_iou(bbox1, bbox2):
    region1 = (bbox1[0], bbox1[1], bbox1[0]+bbox1[2], bbox1[1]+bbox1[3])
    region2 = (bbox2[0], bbox2[1], bbox2[0] + bbox2[2], bbox2[1] + bbox2[3])
    cross_w = min(region1[2], region2[2]) - max(region1[0], region2[0])
    cross_h = min(region1[3], region2[3]) - max(region1[1], region2[1])

    if cross_w <= 0 or cross_h <= 0:
        cross = 0
    else:
        cross = cross_w * cross_h
    s1 = bbox1[2] * bbox1[3]
    s2 = bbox2[2] * bbox2[3]
    iou = float(cross)/float(s1 + s2 - cross)
    return max(iou, 1e-5)


def hungarian(matrix):
    m = Munkres()
    matches = m.compute(matrix)
    return matches


class Obstacle():
    count = -1


class Track():
    cur_frame = -1
    id = -1
    is_missing = False
    missing_pred = []  # Prediction around the obstacle when missing.


class Detection():
    default = -1
    id = -1

