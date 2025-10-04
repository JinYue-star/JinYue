#!/usr/bin/python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import jetson.inference
import jetson.utils
import numpy as np
import argparse
import sys
import cv2

parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, epilog=jetson.inference.poseNet.Usage() +
                                 jetson.utils.videoSource.Usage() + jetson.utils.videoOutput.Usage() + jetson.utils.logUsage())

parser.add_argument("input_URI", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output_URI", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="none", help="Disable default skeleton overlay to avoid duplication")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	opt = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

net = jetson.inference.poseNet(opt.network, sys.argv, opt.threshold)

input = jetson.utils.videoSource(opt.input_URI, argv=sys.argv)
output = jetson.utils.videoOutput(opt.output_URI, argv=sys.argv)

def calculate_angle(pose, id1, id2, id3):
    point1_x = np.array(pose.Keypoints[id1].x)
    point1_y = np.array(pose.Keypoints[id1].y)
    point2_x = np.array(pose.Keypoints[id2].x)
    point2_y = np.array(pose.Keypoints[id2].y)
    point3_x = np.array(pose.Keypoints[id3].x)
    point3_y = np.array(pose.Keypoints[id3].y)
    v1 = (point1_x - point2_x, point1_y - point2_y)
    v2 = (point3_x - point2_x, point3_y - point2_y)
    cosine_angle = np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

KEYPOINT_COLOR = (0, 255, 0)
KEYPOINT_RADIUS = 5
LINE_COLOR = (255, 0, 0)
LINE_THICKNESS = 3
TEXT_COLOR = (255, 255, 255)
TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SIZE = 0.6
TEXT_THICKNESS = 2

while True:
    img_jetson = input.Capture()

    img_rgb = jetson.utils.cudaToNumpy(img_jetson)
    img_rgb = img_rgb[:, :, :3]
    img_opencv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    poses = net.Process(img_jetson, overlay=opt.overlay)

    print("detected {:d} objects in image".format(len(poses)))

    for pose in poses:
        print("------------------------------------------------------------------------------")
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)
        
        target_keypoints = {
            'left_shoulder': pose.FindKeypoint('left_shoulder'),
            'left_elbow': pose.FindKeypoint('left_elbow'),
            'left_wrist': pose.FindKeypoint('left_wrist')
        }
        idx_1 = target_keypoints['left_shoulder']
        idx_2 = target_keypoints['left_elbow']
        idx_3 = target_keypoints['left_wrist']
        if idx_1 < 0 or idx_2 < 0 or idx_3 < 0:
            print("Warning: Left arm keypoints not fully detected, skipping...")
            continue
        
        angle = calculate_angle(pose, idx_1, idx_2, idx_3)
        print(f"Angle of Left Arm (left_shoulder-left_elbow-left_wrist): {angle:.2f}deg")
        print("------------------------------------------------------------------------------")

        s_x = int(pose.Keypoints[idx_1].x)
        s_y = int(pose.Keypoints[idx_1].y)
        e_x = int(pose.Keypoints[idx_2].x)
        e_y = int(pose.Keypoints[idx_2].y)
        w_x = int(pose.Keypoints[idx_3].x)
        w_y = int(pose.Keypoints[idx_3].y)

        cv2.circle(img_opencv, (s_x, s_y), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)
        cv2.circle(img_opencv, (e_x, e_y), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)
        cv2.circle(img_opencv, (w_x, w_y), KEYPOINT_RADIUS, KEYPOINT_COLOR, -1)

        cv2.line(img_opencv, (s_x, s_y), (e_x, e_y), LINE_COLOR, LINE_THICKNESS)
        cv2.line(img_opencv, (e_x, e_y), (w_x, w_y), LINE_COLOR, LINE_THICKNESS)

        text = f"Left Arm Angle: {angle:.2f}deg"
        # Move text 30 pixels left and 30 pixels up from left elbow (adjusted position)
        text_pos = (e_x - 30, e_y - 30)
        cv2.putText(img_opencv, text, text_pos, TEXT_FONT, TEXT_SIZE, TEXT_COLOR, TEXT_THICKNESS)

    img_rgb_back = cv2.cvtColor(img_opencv, cv2.COLOR_BGR2RGB)
    img_rgba_back = np.dstack((img_rgb_back, np.ones_like(img_rgb_back[:, :, 0]) * 255))
    img_jetson_back = jetson.utils.cudaFromNumpy(img_rgba_back)

    output.Render(img_jetson_back)
    output.SetStatus("{:s} | Network {:.0f} FPS".format(opt.network, net.GetNetworkFPS()))
    net.PrintProfilerTimes()

    if not input.IsStreaming() or not output.IsStreaming():
        break