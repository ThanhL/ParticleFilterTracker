### Libraries
import cv2
import time
import argparse
import numpy as np
from YOLO_CV_Wrapper import YOLO_CV_Wrapper
from Particles import Particles
from ParticleFilter import *
from MultiParticleFilterTracker import MultiParticleFilterTracker

# Set seed for repeatability
np.random.seed(seed=200)

## Global Parameters
DEFAULT_VIDEO = "./datasets/TownCenter.mp4"
# DEFAULT_VIDEO = "./datasets/single_white_car.mp4"
# DEFAULT_VIDEO = "./datasets/volkswagen_pikes_peak_original_cut_3.mp4"
# DEFAULT_VIDEO = "./datasets/MOT20-02-raw.webm"

YOLOV3_SPP_WEIGHTS = "./models/yolov3/yolov3-spp.weights"
YOLOV3_SPP_CFG = "./models/yolov3/yolov3-spp.cfg"
YOLOV3_WEIGHTS = "./models/yolov3/yolov3.weights"
YOLOV3_CFG = "./models/yolov3/yolov3.cfg"
COCO_CLASSES_TXT = "./models/yolov3/coco.names"
YOLO_CONFIDENCE_THRESH = 0.6

## Color matrix for each unique track
MAX_TRACKS = 300
TRACK_COLORS = [tuple(np.random.randint(0,255,(1,3), dtype="int").squeeze()) for i in range(MAX_TRACKS)]

## Particle filter params
N = 200                                                 # Number of particles per object
initial_estimate_covariance = np.array([4, 2, 4, 2])    # 

### Particles Opencv utility funcs
def draw_particles(frame, particles, color=(255,255,0)):
    for particle in particles:
        x, x_dot, y, y_dot = particle
        cv2.circle(frame, (int(x), int(y)), 2, color)
    return frame

def draw_particles_mean(frame, particles, weights):
    ## Get mean and covariance from particles and their respective weights
    particles_mean, particles_covariance = estimate_particles(particles, weights)
    x, x_dot, y, y_dot = particles_mean

    ## Draw with the calculated mean
    cv2.circle(frame, (int(x), int(y)), 4, (255,255,255), -1)
    return frame

def draw_pf_track(frame, pf_track, color=(255,255,0)):
    ### Draws track estimates from particles and their id
    ## Draw all particles
    frame = draw_particles(frame, pf_track.particles, color=color)

    ## Draw position estimate of particle set
    # First get the mean estimate and extract states
    particles_mean, particles_covariance = estimate_particles(pf_track.particles, pf_track.weights)
    x, x_dot, y, y_dot = particles_mean

    # Draw the calculated mean
    cv2.circle(frame, (int(x), int(y)), 4,  (255,255,255), -1)

    ## Draw track ID
    cv2.putText(frame, "ID: {}".format(str(pf_track.trackID)), (int(x), int(y-20)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame


### Driver main
def main():
    ### Argument parser setup
    parser = argparse.ArgumentParser(description='particle filter tracker')
    parser.add_argument('--yolo_weights', dest='yolo_weights', default=YOLOV3_WEIGHTS,
                        help='yolo pretrained weights to be used by particle filter')
    parser.add_argument('--yolo_cfg', dest='yolo_cfg', default=YOLOV3_CFG,
                        help='yolo cfg to be used by particle filter')    
    parser.add_argument('--yolo')
    parser.add_argument('--video', dest='video', default=DEFAULT_VIDEO,
                        help='video to track object')
    parser.add_argument('--gpu', type=bool, default=False)
    parser.add_argument('--debug_cv', default=False, action="store_true")
    parser.add_argument('--output', dest='output', default=None)


    args = parser.parse_args()
    yolo_weights = args.yolo_weights
    yolo_cfg = args.yolo_cfg
    video = args.video
    gpu_enable = args.gpu
    debug_cv = args.debug_cv
    output_video = args.output

    ### YOLO Wrapper Creation
    yolo_cv = YOLO_CV_Wrapper(yolo_weights=yolo_weights,
                            yolo_cfg=yolo_cfg,
                            yolo_classes=COCO_CLASSES_TXT,
                            gpu_enabled=gpu_enable,
                            confidence_thresh=YOLO_CONFIDENCE_THRESH)

    ### Particle Filter Tracker Creation
    multi_pf_tracker = MultiParticleFilterTracker(N)


    ### Run the tracker on video
    cap = cv2.VideoCapture(video)

    # Extract first frame details
    ret, frame = cap.read()

    if output_video:
        print("[!] Writting to video file: {}".format(output_video))

        # Define the codec and create VideoWriter object for saving video
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_video),fourcc, 20.0, (frame.shape[1],frame.shape[0]))

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Pass frame through yolo
        detections = yolo_cv.detect_objects(frame)

        for detection in detections:
            x, y, w, h = detection["boxes"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)


        ### Update particles with detection
        multi_pf_tracker.update_particle_tracks(detections)


        ### Draw particles onto frame
        for pf_track in multi_pf_tracker.particle_tracks:
            # Draw particle filter track containing particles for the track, mean position and ID number
            frame = draw_pf_track(frame, pf_track, color=(int(TRACK_COLORS[pf_track.trackID][0]),
                                                        int(TRACK_COLORS[pf_track.trackID][1]),
                                                        int(TRACK_COLORS[pf_track.trackID][2])))

        ### Display the resulting frame
        cv2.imshow('frame',frame)

        ### Save frame to video
        if output_video:
            out.write(frame)
        
        ### Wait for key press if debug enabled
        if debug_cv:
            key = cv2.waitKey(0)
            while key not in [ord('q'), ord('k')]:
                key = cv2.waitKey(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
            


if __name__ == "__main__":
    main()