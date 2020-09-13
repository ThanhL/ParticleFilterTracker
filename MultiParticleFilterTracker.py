import numpy as np
from Particles import Particles
from ParticleFilter import *
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


class ParticleTrack():
    def __init__(self, N, trackID, process_noise=[0.2,2,0.2,2], sensor_noise=2):
        self.N = N
        self.particles = np.empty((N,4))
        self.weights = np.ones(N) / N
        self.trackID = trackID 
        self.track_strikes = 0
        self.process_noise = process_noise
        self.sensor_noise = sensor_noise

        # State estimates
        self.state_estimate = np.zeros(4)
        self.state_position_estimate = np.zeros(2)

    ### Particle filter step for this track
    def pf_step(self, detection, dt=0.2):
        ### Particle Filter Predict Step
        self.particles = pf_predict(self.particles, 
                                    dt, 
                                    process_noise=self.process_noise)

        ### Particle Filter Update Step
        # Get particles position estiamte
        track_mean, track_cov = estimate_particles(self.particles, self.weights)
        track_position = np.array([track_mean[0], track_mean[1]])

        # Get measurement of tracked object w.r.t to detection
        dist_from_det_to_mean = np.linalg.norm(track_position - detection)

        self.weights = pf_update(self.particles, 
                                self.weights, 
                                dist_from_det_to_mean, 
                                sensor_noise=self.sensor_noise, 
                                detection_landmark=detection)

        # Resample if we drop below the number of effective particles
        if neff(self.weights) < (len(self.particles) / 2):
            # Resample with stratified resample
            indexes = stratified_resample(self.weights)

            self.particles, self.weights = resample_from_index(self.particles, self.weights, indexes)
            assert np.allclose(self.weights, 1/self.N) 


    ## Particles initializer
    def create_uniform_particles(self, x_range, x_dot_range, y_range, y_dot_range):
        # Creates a set of particles using uniform distribution
        particles = np.empty((self.N, 4))
        particles[:,0] = np.random.uniform(x_range[0], x_range[1], self.N)           # x_state
        particles[:,1] = np.random.uniform(x_dot_range[0], x_dot_range[1], self.N)   # x_dot state
        particles[:,2] = np.random.uniform(y_range[0], y_range[1], self.N)           # y_state
        particles[:,3] = np.random.uniform(y_dot_range[0], y_dot_ange[1], self.N) 
        return particles

    def create_gaussian_particles(self, mu, std):
        # Takes in mean and covariance and outputs randomly distributed particles
        particles = np.empty((self.N, 4))
        particles[:,0] = np.random.normal(mu[0], std[0], self.N)     # x_state
        particles[:,1] = np.random.normal(mu[1], std[1], self.N)     # x_dot state
        particles[:,2] = np.random.normal(mu[2], std[2], self.N)     # y_state
        particles[:,3] = np.random.normal(mu[3], std[3], self.N)     # y_dot state
        return particles



class MultiParticleFilterTracker():
    def __init__(self, N, euclidean_dist_thresh=50, max_track_strikes=10):
        # Particle set for each unique object
        self.particle_tracks = []

        # Number of particles to use across all sets
        self.N = N

        # Initial states
        # Note: we don't need to set initial x or y as the detection centers will be the initial values for these
        self.initial_estimate_covariance = np.array([2, 4, 2, 4])   # Initial estimate covarianc
        self.initial_x_dot = 2  # Initial x velocity
        self.initial_y_dot = 2  # Initial y velocity

        # Gaussian process noise for states x, x_dot, y, y_dot
        self.process_noise = np.array([1.5,2,1.5,2])
        # self.process_noise = np.array([2.5,4.2,2.5,4.2])

        # Sensor noise for detection measurements from yolo
        self.sensor_noise = 4   # 1-3 seems to be good

        # ID count
        self.id_count = 0

        # Euclidean distance threshold for detection centers w.r.t particle estimates
        self.euclidean_dist_thresh = euclidean_dist_thresh

        # Max skipped detections
        self.max_track_strikes = max_track_strikes


    def extract_position_estimates(self):
        ## Extract position estimates from all particle sets so they can 
        ## be used for hungarian assignment's cost matrix
        all_track_mean_postion = []
        for pf_track in self.particle_tracks:
            track_mu, track_cov = estimate_particles(pf_track.particles, pf_track.weights)
            track_position = np.array([track_mu[0], track_mu[2]])
            all_track_mean_postion.append(track_position)
        all_track_mean_postion = np.array(all_track_mean_postion)              
        return all_track_mean_postion

    def extract_detection_centers(self, detections):
        ## Extract all detection centers so they can be used for hungarian assignment cost matrix
        detection_center_positions = []
        for detection in detections:
            detection_center = detection["center"]
            detection_center_positions.append(detection_center)
        detection_center_positions = np.array(detection_center_positions)
        return detection_center_positions


    def update_particle_tracks(self, detections, dt=0.5):
        ### Predict step
        for pf_track in self.particle_tracks:
            pf_track.particles = pf_predict(pf_track.particles, dt, process_noise=self.process_noise)

        ### Update particles track with measurements
        if len(detections) != 0:
           
            if len(self.particle_tracks) == 0:
                # Nothing in the particle tracks list, create particle tracks for each detection
                for detection in detections:
                    # Extract centers
                    det_center_x, det_center_y = detection["center"]

                    # Create new particle set
                    new_particle_track = ParticleTrack(self.N, self.id_count)
                    new_particle_track.particles = new_particle_track.create_gaussian_particles(
                                                                    mu=(det_center_x, self.initial_x_dot, det_center_y, self.initial_y_dot),
                                                                    std=self.initial_estimate_covariance)

                    self.particle_tracks.append(new_particle_track)

                    # Update ID counter
                    self.id_count += 1

            else:
                ### Use hungarian assignment to calculate detection particle track pairs
                ## Get the means of each particle set and detection and calculate distance matrix to be used 
                ## in hungarian assignment
                # Extract all track mean position
                all_track_mean_postion = self.extract_position_estimates()

                # Extract detection centers
                detection_center_positions = self.extract_detection_centers(detections) 

                # Calculate euclidean distance matrix with detection centers and particle means
                euclidean_distance_matrix = distance_matrix(all_track_mean_postion, detection_center_positions)

                # Find optimal assignments with hungarian
                row_ind, col_ind = linear_sum_assignment(euclidean_distance_matrix)

                # Get unassigned detections based on the column indexes of hungarian assignment
                unassigned_detections = np.delete(detection_center_positions, col_ind, axis=0)


                # Debugging hungarian assignment
                for i in range(len(col_ind)):
                    track_mu, track_cov = estimate_particles(self.particle_tracks[row_ind[i]].particles,
                                                            self.particle_tracks[row_ind[i]].weights)
                    track_position = (track_mu[0], track_mu[2])

                    print("Track {}: {}".format(self.particle_tracks[row_ind[i]].trackID, 
                                            track_position))
                    print("{} --> {}".format(all_track_mean_postion[row_ind[i]], detection_center_positions[col_ind[i]]))
                    print("Euclidean Dist from matrix: {}".format(euclidean_distance_matrix[row_ind[i]][col_ind[i]]))
                    print("Euclidean Dist from calculation: {}".format(
                        np.linalg.norm(track_position - detection_center_positions[col_ind[i]])))
                    print()

                # Debug tracks
                for i in range(len(self.particle_tracks)):
                    print("Particle Track ID: {} \t Strikes: {}".format(self.particle_tracks[i].trackID, 
                                                                    self.particle_tracks[i].track_strikes))
                
                ### For each assigned detection, update the corresponding particle sets
                ## Particle sets should be in the order with the assigned_detections
                for i in range(len(col_ind)):
                    # Check if euclidean distance satisfies threshold
                    if euclidean_distance_matrix[row_ind[i]][col_ind[i]] <= self.euclidean_dist_thresh:
                        ### Accept this measurement and update the particles 
                        ## Reset strike counter
                        self.particle_tracks[row_ind[i]].track_strikes = 0
                    
                        ## Calculate the euclidean distance of particles mean position estimate w.r.t measurements center                        
                        dist_from_det_to_mean = np.linalg.norm(all_track_mean_postion[row_ind[i]] - detection_center_positions[col_ind[i]])
                        
                        ## Update the corresponding particle weights
                        self.particle_tracks[row_ind[i]].weights = pf_update(self.particle_tracks[row_ind[i]].particles, 
                                                                        self.particle_tracks[row_ind[i]].weights, 
                                                                        dist_from_det_to_mean, 
                                                                        sensor_noise=self.sensor_noise, 
                                                                        detection_landmark=detection_center_positions[col_ind[i]])

                        ## Resample if we drop below the number of effective particles
                        if neff(self.particle_tracks[row_ind[i]].weights) < (len(self.particle_tracks[row_ind[i]].particles) / 2):
                            # # Resample with resampling wheel
                            # self.particle_tracks[row_ind[i]].particles = resampling_wheel(self.particle_tracks[row_ind[i]].particles, 
                            #                                                     self.particle_tracks[row_ind[i]].weights)

                            # Resample with staatified resample
                            indexes = stratified_resample(self.particle_tracks[row_ind[i]].weights)

                            self.particle_tracks[row_ind[i]].particles, self.particle_tracks[row_ind[i]].weights = resample_from_index(
                                self.particle_tracks[row_ind[i]].particles, self.particle_tracks[row_ind[i]].weights, indexes)

                            # Ensure after resampling all weights are the same for this particle distribution
                            assert np.allclose(self.particle_tracks[row_ind[i]].weights, 1./self.N)
                    
                    else:
                        ### Reject this measurement and add a strike
                        self.particle_tracks[row_ind[i]].track_strikes += 1


                ### For each particle track, if the particles haven't been updated after max strikes then remove from set
                for pf_track in self.particle_tracks:
                    if pf_track.track_strikes >= self.max_track_strikes:
                        print("Track {} removed!".format(pf_track.trackID))
                        self.particle_tracks.remove(pf_track)


                ### For each unassigned detection, create new particle set
                for i in range(len(unassigned_detections)):
                    # Extract centers from unassigned detection
                    det_center_x, det_center_y = unassigned_detections[i]

                    # Create new particle set
                    new_particle_track = ParticleTrack(self.N, self.id_count)
                    new_particle_track.particles = new_particle_track.create_gaussian_particles(
                                                                    mu=(det_center_x, self.initial_x_dot, det_center_y, self.initial_y_dot),
                                                                    std=self.initial_estimate_covariance)

                    # Add new particle set to existing tracks
                    self.particle_tracks.append(new_particle_track)

                    # Update ID counter
                    self.id_count += 1                
                