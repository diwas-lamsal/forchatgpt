# Copyright (c) OpenMMLab. All rights reserved.
from collections import deque
import os
import warnings
from argparse import ArgumentParser

import cv2

from mmpose.apis import (get_track_id, inference_top_down_pose_model,
                         init_pose_model, process_mmdet_results,
                         vis_pose_tracking_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False


# Other imports:
from scipy.signal import butter, filtfilt
from shapely.geometry import Point, Polygon
import time
import numpy as np
from scipy import signal
from ec2.utils import nan_helper
from ec2.classification import infer_cnn1d


def remove_false_positives_using_pose_estimation_results(pose_results, threshold = 0.3, min_keypoints = 4):
    if len(pose_results)>0:
        del_list = []
        for i in range(len(pose_results)):
            pr = pose_results[i]
            below_threshold = len(pr['keypoints'][pr['keypoints'][:,2] < threshold]) 
            if below_threshold>min_keypoints:
                del_list.append(i)
        del_list=sorted(del_list, key=int, reverse=True)
        for d in del_list:
            del pose_results[d]
    return pose_results


def main():
    """Visualize the demo images.

    Using mmdet to detect the human.
    """
    parser = ArgumentParser()
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--video-path', type=str, help='Video path')
    parser.add_argument(
        '--classifier',
        default="/root/EC2/demo/ec2/models/vitpose_1dcnn_obj_det_fine_tuned_fallv4.1.2.pth"
    )
    parser.add_argument(
        '--show',
        action='store_true',
        default=False,
        help='whether to show visualizations.')
    parser.add_argument(
        '--out-video-root',
        default='',
        help='Root of the output video file. '
        'Default not saving the visualization video.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--det-cat-id',
        type=int,
        default=1,
        help='Category id for bounding box detection model')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=0.3,
        help='Bounding box score threshold')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--use-oks-tracking', action='store_true', help='Using OKS tracking')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.2, help='Tracking threshold')
    parser.add_argument(
        '--euro',
        action='store_true',
        help='Using One_Euro_Filter for smoothing')
    parser.add_argument(
        '--radius',
        type=int,
        default=4,
        help='Keypoint radius for visualization')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization')

    assert has_mmdet, 'Please install mmdet to run the demo.'

    args = parser.parse_args()

    assert args.show or (args.out_video_root != '')
    assert args.det_config is not None
    assert args.det_checkpoint is not None

    det_model = init_detector(
        args.det_config, args.det_checkpoint, device=args.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.pose_config, args.pose_checkpoint, device=args.device.lower())

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    cap = cv2.VideoCapture(args.video_path)
    fps = None

    assert cap.isOpened(), f'Faild to load video file {args.video_path}'

    if args.out_video_root == '':
        save_out_video = False
    else:
        os.makedirs(args.out_video_root, exist_ok=True)
        save_out_video = True

    # Resize window to 1080p
    if args.show:
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL )
        cv2.resizeWindow('Output', 1920,1080)

    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # for 1080p
        size = (int(1920),
                int(1080))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(
            os.path.join(args.out_video_root,
                         f'vis_{os.path.basename(args.video_path)}'), fourcc,
            fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    next_id = 0
    pose_results = []


    # Added from main-eldercare:
    prev_frame_time = 0

    framecount = 0
    filtered_framcount=0
    buffercount=10

    number_joints=18
    posefilter=True
    filtertype="lowpass"

    pose_results_buffer=deque([], maxlen=buffercount)
    pose_results_buffer_list=[]
    pose_results_buffer_list_update=0
    pose_results_buffer_list_filtered_joints=[]
    id_joints={}

    lstm_posedict={}

    classify=True
    classify_model="cnn1d"  #lstm
    lstm_framecount=28
    avoidkp=[1,2,3,4]

    frame_buffer=deque([], maxlen=buffercount)

    # classification_model = infer_cnn1d("/home/diwas/AICenter/Extract/ViTPose/demo/ec2/models/vitpose_1dcnn_propelled_v2.1.pth")
    classification_model = infer_cnn1d(args.classifier)

    if posefilter==True:
        if filtertype=="lowpass":
            print("Filter lowpass")
            fs = 18  # Sampling frequency
            fc = 3  # Cut-off frequency of the filter
            wn = fc / (fs / 2) # Normalize the frequency
            butter_b, butter_a = butter(2, wn)
        else:
            print("Filter salvago")
            window_length, polyorder = 9, 2   #salvago filter

    fpscount=0

    action_buffer={}
    fall_buffer={}
    fall_saved=set()  # count saved track_id

    def refine_classification(result,track_id):
        if track_id in action_buffer:   
            if result in action_buffer[track_id]:
                action_buffer[track_id][result] = action_buffer[track_id][result] + 1
            else:
                action_buffer[track_id] = {result : 1}
        else:
            action_buffer[track_id] = {result : 1}
        
        if action_buffer[track_id][result]>15:   # only if 15 consective action classification, then send True
            return True
        else:
            return False        




    while (cap.isOpened()):
        x_pos_action = 300

        start_t = time.time() # FPS

        pose_results_last = pose_results

        flag, img = cap.read()

        if not flag:
            break

        img = cv2.resize(img, dsize=(1920, 1080))#, interpolation=cv2.INTER_AREA)
        # img = cv2.resize(img, dsize=(640, 360))#, interpolation=cv2.INTER_AREA)

        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, args.det_cat_id)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=args.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)


        # Del pose results if most keypoints below a certain threshold
        pose_results=remove_false_positives_using_pose_estimation_results(pose_results, threshold=0.3, min_keypoints=7)

        # get track id for each person instance
        pose_results, next_id = get_track_id(
            pose_results,
            pose_results_last,
            next_id,
            use_oks=args.use_oks_tracking,
            tracking_thr=args.tracking_thr,
            use_one_euro=args.euro,
            fps=fps)
        

        # ec2 code from here
        if posefilter:
            if framecount>=buffercount:

                if pose_results_buffer_list_update==0:
                    pose_results_buffer_list_filtered_joints=[]
                    pose_results_buffer_list=list(pose_results_buffer)

                    tracked_id = {}

                    for frame in pose_results_buffer_list:
                        for person in frame:
                            if person['track_id'] in tracked_id:
                                tracked_id[person['track_id']] = tracked_id[person['track_id']]+ 1
                            else:
                                tracked_id[person['track_id']] = 0


                    id_joints={}
                    for id in tracked_id.keys():
                        if tracked_id[id] < 2 or id==-1:   # if trackedid==-1 remove track, if tracked instances less then 2, remove track
                            continue

                        id_joints[id] = [ [0]*buffercount for n in range(number_joints*2) ]


                    frameidx=0
                    for frame in pose_results_buffer_list:
                        for person in frame:

                            person_id=person['track_id']
                            if person_id not in id_joints:
                                continue

                            idx=0

                            for kp in person['keypoints']:

                                id_joints[person_id][idx][frameidx]=kp[0]
                                id_joints[person_id][idx + 1][frameidx]=kp[1]

                                idx=idx+2

                        frameidx = frameidx + 1

                    filter_joint = []

                    for track_id in id_joints.keys():
                        jidx = 0
                        for joint in id_joints[track_id]:
                            if not any(joint):
                                filter_joint=joint
                            else:
                                interp_joint = np.array(joint)

                                # interpolate
                                interp_joint[interp_joint == 0] = np.nan
                                nans, x = nan_helper(interp_joint)
                                interp_joint[nans] = np.interp(x(nans), x(~nans), interp_joint[~nans])

                                if filtertype == "lowpass":
                                    filter_joint = filtfilt(butter_b, butter_a, interp_joint)
                                else:
                                    filter_joint = signal.savgol_filter(interp_joint, window_length, polyorder)


                            id_joints[track_id][jidx]=filter_joint
                            jidx = jidx + 1

                    pose_results_buffer_list_update=pose_results_buffer_list_update+1


                frameidx=pose_results_buffer_list_update-1

                current_joint=[]

                values=0
                for track_id in id_joints.keys():
                    format_joint=[]

                    for j in range(0,number_joints*2,2):
                        format_joint.append([id_joints[track_id][j][frameidx],id_joints[track_id][j+1][frameidx]])

                    if any(format_joint):
                        if classify:
                            lstm_joints=[]
                            jcount=0
                            for j in format_joint:
                                if(j[0]==0 and j[1]==0):
                                    continue    
                                lstm_joints.extend([j[0],j[1]])
                                jcount=jcount+1

                            if track_id in lstm_posedict:
                                lstm_posedict[track_id].append(lstm_joints)
                            else:
                                lstm_posedict[track_id] = deque([], maxlen=lstm_framecount)  #TODO: lstm_posedict does not clean up old track_id pose data, !! mem leak !!
                                lstm_posedict[track_id].append(lstm_joints)


                            if len(lstm_posedict[track_id])==lstm_framecount:
                                a_input = np.array([list(lstm_posedict[track_id])], dtype=np.float32)
                                class_time = time.time()
                                out, out2, out3 = classification_model.infer(a_input, return_secondary=True)
                                
                                for person in pose_results_buffer_list[-1]:
                                    if person['track_id']==track_id:
                                        xy_coords = person['bbox'][:2]

                                cv2.putText(img, out+", Track id:"+str(track_id), (int(xy_coords[0]),int(xy_coords[1])), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                                                                
                                # For showing second and third actions
                                # cv2.putText(img, f'Second prediction:{out2}', (25,x_pos_action+40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                                # cv2.putText(img, f'Third prediction:{out3}', (25,x_pos_action+80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)

                                x_pos_action+=250

                                current_joint.extend(format_joint)
                            else:
                                current_joint.extend(format_joint)
                        else:
                            current_joint.extend(format_joint)


                filtered_framcount=filtered_framcount+1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                pose_results_buffer_list_update=pose_results_buffer_list_update+1

                if pose_results_buffer_list_update==buffercount+1:
                    pose_results_buffer_list_update=0

                pose_results_buffer.popleft()
                prev_pos_results = pose_results.copy()
                pose_results_buffer.append(prev_pos_results)
                frame_buffer.append(img)

            else:
                prev_pos_results = pose_results.copy()
                pose_results_buffer.append(prev_pos_results)
                frame_buffer.append(img)

        # show the results
        vis_img = vis_pose_tracking_result(
            pose_model,
            img,
            pose_results,
            radius=args.radius,
            thickness=args.thickness,
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=args.kpt_thr,
            show=False)
        

        if args.show:
            cv2.imshow('Output', vis_img)

        if save_out_video:
            videoWriter.write(vis_img)

        if args.show and cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        framecount=framecount+1
        if (framecount%30)==0:
           print("frame", framecount)
           fps_measure=1/(time.time()-start_t)
           print("fps time ", fps_measure)
           fpscount=fpscount+fps_measure

    cap.release()
    if save_out_video:
        videoWriter.release()
    if args.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
