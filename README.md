# Multi Object Tracking with Particle Filters, Hungarian Assignment and YOLO

## Repository + Algorithm Summary 

This repository implements a multiple object tracker using particle filters in combination with YOLO. Each object have their own "set" of particles that are updated with the detections of YOLO. These YOLO detections act as the measurements for the particle filter tracking. To determine which detection belongs to which particle set, the hungarian assignment is used to assign the detections from YOLO to the appropiate particle set and update them accordingly. New particle sets are initialized for unassigned detections. 


## Algorithm Pseudocode

The particle filter is responsible for tracking the positions (x,y) and the velocity (x_dot, y_dot) of the objects being tracked. Each unique object contains a set of particles representing the object's particle distribution. 

YOLO detection outputs are used as measurement updates for the particle filter algorithm. To match these detections

## Setup/Installation

This project was built and tested with Python 3.8.5. Has not been tested with Python 2.7. The required packages and their versions are located in the requirements.txt file. 

To run this project, first clone the repository and install the required python packages with the requirements.txt:

```
$ cd <directory you want to install to>
$ git clone https://github.com/ThanhL/ParticleFilterTracker.git
$ cd ParticleFilterTracker
$ pip install -r requirements.txt
```

Since this particle filter tracker uses YOLO detection as it measurements when tracking objects, we first need to download the following files and store them in the models folder. 

* **YOLO weights**: File containing the pre-trained network's weights.
* **YOLO cfg**: File containing the network configuration.
* **coco.names**: File containing the 80 different class names used in COCO dataset. coco.names can be downloaded [here](https://github.com/pjreddie/darknet/blob/master/data/coco.names)

Note the code was built and tested with YOLO-v3 SPP architecture. The cfg and pretrained YOLO weights can be downloaded from  https://pjreddie.com/darknet/yolo/ unser *Performance on the COCO Dataset*.

## Basic Use

To use the multi o





## References
[1] YOLO object detection with OpenCV, Adrian Rosebrock. https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/

[2] Kalman and Bayesian Filters in Python - Particle Filters, Roger Labbe. https://github.com/rlabbe/
Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb

[3] Multi Object Tracking Tutorial, Student Dave. https://www.youtube.com/watch?v=Me0wbxEDO4I&feature=youtu.be

[4] Thrun, Sebastian., Burgard, Wolfram., Fox, Dieter. *Probablistic Robotics*, 2005.

[5] Hungarian algorithm, Wikipedia. https://en.wikipedia.org/wiki/Hungarian_algorithm