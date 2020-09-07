import cv2
import time
import argparse
import numpy as np
from YOLO_CV_Wrapper import YOLO_CV_Wrapper

DEFAULT_VIDEO = "./datasets/single_white_car.mp4"
YOLOV3_SPP_WEIGHTS = "./models/yolov3/yolov3-spp.weights"
YOLOV3_SPP_CFG = "./models/yolov3/yolov3-spp.cfg"
YOLOV3_WEIGHTS = "./models/yolov3/yolov3.weights"
YOLOV3_CFG = "./models/yolov3/yolov3.cfg"
COCO_CLASSES_TXT = "./models/yolov3/coco.names"
YOLO_CONFIDENCE_THRESH = 0.5


### YOLO utility funcs


### Driver main
def main():
    ### Argument parser setup
    parser = argparse.ArgumentParser(description='particle filter tracker')
    parser.add_argument('--yolo_model', dest='yolo_model', default="None",
                        help='yolo model to be used by particle filter')
    parser.add_argument('--video', dest='video', default=DEFAULT_VIDEO,
                        help='video to track object')
    parser.add_argument('--gpu', type=bool, default=False)


    args = parser.parse_args()
    yolo_model = args.yolo_model
    video = args.video
    gpu_enable = args.gpu


    ### YOLO Wrapper Creation
    yolo_cv = YOLO_CV_Wrapper(yolo_weights=YOLOV3_WEIGHTS,
                            yolo_cfg=YOLOV3_CFG,
                            yolo_classes=COCO_CLASSES_TXT,
                            gpu_enabled=gpu_enable,
                            confidence_thresh=YOLO_CONFIDENCE_THRESH)


    ### Run the tracker
    cap = cv2.VideoCapture(video)


    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Pass frame through yolo
        frame = yolo_cv.darknet_process_frame(frame)

        # Display the resulting frame
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
            


if __name__ == "__main__":
    main()