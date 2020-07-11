# To use Inference Engine backend, specify location of plugins:
# export LD_LIBRARY_PATH=/opt/intel/deeplearning_deploymenttoolkit/deployment_tools/external/mklml_lnx/lib:$LD_LIBRARY_PATH

import cv2
import numpy as np
import argparse
import imutils
import time
# import freenect

parser = argparse.ArgumentParser(
        description='This script is used to demonstrate OpenPose human pose estimation network '
                    'from https://github.com/CMU-Perceptual-Computing-Lab/openpose project using Opencv2. '
                    'The sample and model are simplified and could be used for a single person on the frame.')
# parser.add_argument('--input', help='Path to input image.')
parser.add_argument('--proto', help='Path to .prototxt')
parser.add_argument('--model', help='Path to .caffemodel')
parser.add_argument('--dataset', help='Specify what kind of model was trained. '
                                      'It could be (COCO, MPI) depends on dataset.')
parser.add_argument('--thr', default=0.1, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]



net = cv2.dnn.readNetFromCaffe("pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt", "pose/mpi/pose_iter_160000.caffemodel")

cap = cv2.VideoCapture(0)

# frame = cv2.imread(args.input)
__, frame = cap.read()
#frame = imutils.rotate(frame, 270)
#frame = frame.transpose((1,0,2))
# (frame, _) = freenect.sync_get_video()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

inWidth = frameWidth
inHeight = frameHeight

inp = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)
net.setInput(inp)
start_t = time.time()
out = net.forward()
    
print("time is ",time.time()-start_t)

points = []
for i in range(len(BODY_PARTS)):
    # Slice heatmap of corresponging body's part.
    heatMap = out[0, i, :, :]

    # Originally, we try to find all the local maximums. To simplify a sample
    # we just find a global one. However only a single pose at the same time
    # could be detected this way.
    _, conf, _, point = cv2.minMaxLoc(heatMap)
    x = (frameWidth * point[0]) / out.shape[3]
    y = (frameHeight * point[1]) / out.shape[2]

    # Add a point if it's confidence is higher than threshold.
    points.append((int(x), int(y)) if conf > 0.1 else None)

for pair in POSE_PAIRS:
    partFrom = pair[0]
    partTo = pair[1]
    assert(partFrom in BODY_PARTS)
    assert(partTo in BODY_PARTS)

    idFrom = BODY_PARTS[partFrom]
    idTo = BODY_PARTS[partTo]
    if points[idFrom] and points[idTo]:
        cv2.line(frame, points[idFrom], points[idTo], (255, 74, 0), 3)
        cv2.ellipse(frame, points[idFrom], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
        cv2.ellipse(frame, points[idTo], (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
#        cv2.putText(frame, str(idFrom), points[idFrom], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
#        cv2.putText(frame, str(idTo), points[idTo], cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
       
t, _ = net.getPerfProfile()
freq = cv2.getTickFrequency() / 1000
cv2.putText(frame, '%.2fms' % (t / freq), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255),2,cv2.LINE_AA)
if points[5][1]> points[7][1]: 
    print("hand raised") 
else: 
    print("Not raised") 


cv2.imshow("frame", frame)
#cv2.waitKey(1000)
#cv2.imwrite('result_'+args.input,frame)

if cv2.waitKey(0) &0xFF == ord('q'):
    exit