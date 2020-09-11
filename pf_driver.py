import cv2
import time
import argparse
import numpy as np
from YOLO_CV_Wrapper import YOLO_CV_Wrapper
from Particles import Particles
from ParticleFilter import *
from MultiParticleFilterTracker import MultiParticleFilterTracker


DEFAULT_VIDEO = "./datasets/TownCenter.mp4"
YOLOV3_SPP_WEIGHTS = "./models/yolov3/yolov3-spp.weights"
YOLOV3_SPP_CFG = "./models/yolov3/yolov3-spp.cfg"
YOLOV3_WEIGHTS = "./models/yolov3/yolov3.weights"
YOLOV3_CFG = "./models/yolov3/yolov3.cfg"
COCO_CLASSES_TXT = "./models/yolov3/coco.names"
YOLO_CONFIDENCE_THRESH = 0.7

## Particle filter params
N = 100                                                 # Number of particles per object
initial_estimate_covariance = np.array([4, 2, 4, 2])    # 

### Particles Opencv utility funcs
def draw_particles(frame, particles):
    for particle in particles:
        x, x_dot, y, y_dot = particle
        cv2.circle(frame, (int(x), int(y)), 2, (255,0,255))
    return frame

def draw_particles_mean(frame, particles, weights):
    ## Get mean and covariance from particles and their respective weights
    particles_mean, particles_covariance = estimate_particles(particles, weights)
    x, x_dot, y, y_dot = particles_mean

    ## Draw with the calculated mean
    cv2.circle(frame, (int(x), int(y)), 2, (255,255,255))
    return frame


### Driver main
def main():
    ### Argument parser setup
    parser = argparse.ArgumentParser(description='particle filter tracker')
    parser.add_argument('--yolo_model', dest='yolo_model', default="None",
                        help='yolo model to be used by particle filter')
    parser.add_argument('--video', dest='video', default=DEFAULT_VIDEO,
                        help='video to track object')
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--debug_cv', type=bool, default=False)


    args = parser.parse_args()
    yolo_model = args.yolo_model
    video = args.video
    gpu_enable = args.gpu
    debug_cv = args.debug_cv

    ### YOLO Wrapper Creation
    yolo_cv = YOLO_CV_Wrapper(yolo_weights=YOLOV3_WEIGHTS,
                            yolo_cfg=YOLOV3_CFG,
                            yolo_classes=COCO_CLASSES_TXT,
                            gpu_enabled=gpu_enable,
                            confidence_thresh=YOLO_CONFIDENCE_THRESH)

    ### Particle Filter Tracker Creation
    multi_pf_tracker = MultiParticleFilterTracker(N)

    ### Run the tracker on video
    cap = cv2.VideoCapture(video)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Pass frame through yolo
        detections = yolo_cv.detect_objects(frame)
        print(detections)
        for detection in detections:
            print("Boxes: ", detection["boxes"])
            x, y, w, h = detection["boxes"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)


        ### Update particles with detection
        multi_pf_tracker.update_particles(detections)


        ### Draw particles onto frame
        for pf_track in multi_pf_tracker.particle_tracks:
            frame = draw_particles(frame, pf_track.particles)
            frame = draw_particles_mean(frame, pf_track.particles, pf_track.weights)

        ### Display the resulting frame
        cv2.imshow('frame',frame)
        
        ### Wait for key press if debug enabled
        if debug_cv:
            key = cv2.waitKey(0)
            while key not in [ord('q'), ord('k')]:
                key = cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
            


if __name__ == "__main__":
    main()