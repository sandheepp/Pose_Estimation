import cv2
import imutils

cap = cv2.VideoCapture("hand_raise.mp4")
out_video = cv2.VideoWriter('test.avi',cv2.VideoWriter_fourcc(*'DIVX'), 10, (1920, 1080))

status = True

while status:
#for k in range(10):
    status, frame = cap.read()
    frame1 = imutils.rotate(frame,270)
    out_video.write(frame1)

out_video.release()
cap.release()
