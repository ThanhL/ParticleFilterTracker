# Multi Object Tracking with Particle Filters, Hungarian Assignment and YOLO

## Repository + Algorithm Summary 

This repository implements a multiple object tracker using particle filters in combination with YOLO. Each object have their own "set" of particles that are updated with the detections of YOLO. These YOLO detections act as the measurements for the particle filter tracking. To determine which detection belongs to which particle set, the hungarian assignment is used to assign the detections from YOLO to the appropiate particle set and update them accordingly. New particle sets are initialized for unassigned detections. 


## Algorithm Pseudocode

The particle filter is responsible for tracking the positions (x,y) and the velocity (x_dot, y_dot) of the objects being tracked. Each unique object contains a set of particles representing the object's particle distribution. 

YOLO detection outputs are used as measurement updates for the particle filter algorithm. To match these detections

## Usage

## References
* [YOLO object detection with OpenCV](https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/)
 