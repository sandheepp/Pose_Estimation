Code for running OpenPose code for human pose estimation using deep learning in OpenCV. 

First step is download the pretrained models:

# Download the models.
bash get_models.sh


# Resize a Video

ffmpeg -i input.avi -s 1280x720 -aspect 4:3 -vcodec libx264 video_1280x720.avi


# Testing with Webcam:

python3 test.py --proto pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt  --model pose/mpi/pose_iter_160000.caffemodel --dataset MPI


# Custom Command

python3 video_pose.py --input video_720x1280.avi --proto pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt  --model pose/mpi/pose_iter_160000.caffemodel --dataset MPI


# Run the Standard tests

To Run MPI pretrained model on an image sample.jpg:
python3 run_pose.py --input sample.jpg --proto pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt  --model pose/mpi/pose_iter_160000.caffemodel --dataset MPI


To Run Body_25 pretrained model on an image sample.jpg:
python3 run_pose.py --input sample.jpg --proto pose/body_25/body_25_deploy.prototxt  --model pose/body_25/pose_iter_584000.caffemodel


To Run COCO pretrained model on an image sample.jpg:
python3 run_pose.py --input sample.jpg --proto pose/coco/deploy_coco.prototxt  --model pose/coco/pose_iter_440000.caffemodel --dataset COCO




cv2.imshow("frame",frame)
if cv2.waitKey(0) &0xFF == ord('q'):
    exit