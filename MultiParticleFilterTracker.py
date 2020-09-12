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
    def __init__(self, N, euclidean_dist_thresh=30, max_track_strikes=15):
        # Particle set for each unique object
        self.particle_tracks = []

        # Number of particles to use across all sets
        self.N = N

        # Initial estimate covariance
        self.initial_estimate_covariance = np.array([4, 2, 4, 2])

        # Gaussian process noise for states x, x_dot, y, y_dot
        self.process_noise = np.array([2,4,2,4])

        # ID count
        self.id_count = 0

        # Euclidean distance threshold for detection centers w.r.t particle estimates
        self.euclidean_dist_thresh = euclidean_dist_thresh

        # Max skipped detections
        self.max_track_strikes = max_track_strikes


    def update_particle_tracks(self, detections, dt=0.6):
        # ### Predict step
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
                    new_particle_track.particles = new_particle_track.create_gaussian_particles(mu=(det_center_x, 0, det_center_y, 0),
                                                                                                std=self.initial_estimate_covariance)

                    self.particle_tracks.append(new_particle_track)

                    # Update ID counter
                    self.id_count += 1

            else:
                ### Use hungarian assignment to calculate detection particle track pairs
                ## Get the means of each particle set and detection and calculate distance matrix to be used 
                ## in hungarian assignment
                ## TODO: figure out best way computationally to keep track of all the means
                # Extract all track mean position
                all_track_mean_postion = []
                for pf_track in self.particle_tracks:
                    track_mu, track_cov = estimate_particles(pf_track.particles, pf_track.weights)
                    track_position = np.array([track_mu[0], track_mu[2]])
                    all_track_mean_postion.append(track_position)
                all_track_mean_postion = np.array(all_track_mean_postion)              

                # Extract all detections and place in matrix
                detection_center_positions = []
                for detection in detections:
                    detection_center = detection["center"]
                    detection_center_positions.append(detection_center)
                detection_center_positions = np.array(detection_center_positions)


                # Calculate euclidean distance matrix with detection centers and particle means
                euclidean_distance_matrix = distance_matrix(all_track_mean_postion, detection_center_positions)

                # Find optimal assignments with hungarian
                row_ind, col_ind = linear_sum_assignment(euclidean_distance_matrix)

                assigned_detections = np.array([detection_center_positions[col_ind[i]] for i in range(len(col_ind))])
                unassigned_detections = np.delete(detection_center_positions, col_ind, axis=0)


                ### For each assigned detection, update the corresponding particle sets
                ## Particle sets should be in the order with the assigned_detections
                for i in range(len(assigned_detections)):
                    print("all track mean ", i, ": ", all_track_mean_postion[i])
                    print("assigned_detections ", i, ": ", assigned_detections[i])

                    print("euclidean dist: ", euclidean_distance_matrix[i][col_ind[i]])
                    if euclidean_distance_matrix[i][col_ind[i]] <= self.euclidean_dist_thresh:
                        ### Accept this measurement and update the particles 
                        #self.particle_tracks[i].pf_step(detection_center_positions[i])

                        ## Get measurement of tracked object w.r.t to detection and update particle weights
                        dist_from_det_to_mean = np.linalg.norm(all_track_mean_postion[i] - assigned_detections[i])
                        print("Dist from det to mean: ", dist_from_det_to_mean)

                        self.particle_tracks[i].weights = pf_update(self.particle_tracks[i].particles, 
                                                                self.particle_tracks[i].weights, 
                                                                dist_from_det_to_mean, 
                                                                sensor_noise=2, 
                                                                detection_landmark=assigned_detections[i])

                        # Resample if we drop below the number of effective particles
                        if neff(self.particle_tracks[i].weights) < (len(self.particle_tracks[i].particles) / 2):
                            # # Resample with resampling wheel
                            # particles = resampling_wheel(particles, weights)

                            # Resample with staatified resample
                            indexes = stratified_resample(self.particle_tracks[i].weights)

                            self.particle_tracks[i].particles, self.particle_tracks[i].weights = resample_from_index(
                                self.particle_tracks[i].particles, self.particle_tracks[i].weights, indexes)
                            assert np.allclose(self.particle_tracks[i].weights, 1/self.N)

                    else:
                        ### Reject this measurement and add a strike
                        self.particle_tracks[i].track_strikes += 1

                ### For each particle track, if the particles haven't been updated after max strikes then remove from set
                # Debug tracks
                for i in range(len(self.particle_tracks)):
                    print("Particle Track ID: {} \t Strikes: {}".format(self.particle_tracks[i].trackID, 
                                                                    self.particle_tracks[i].track_strikes))

                for pf_track in self.particle_tracks:
                    if pf_track.track_strikes >= self.max_track_strikes:
                        print("Removed!")
                        self.particle_tracks.remove(pf_track)


                ### For each unassigned detection, create new particle set
                for i in range(len(unassigned_detections)):
                    # Extract centers from unassigned detection
                    det_center_x, det_center_y = unassigned_detections[i]

                    # Create new particle set
                    new_particle_track = ParticleTrack(self.N, self.id_count)
                    new_particle_track.particles = new_particle_track.create_gaussian_particles(mu=(det_center_x, 0, det_center_y, 0),
                                                                                                std=self.initial_estimate_covariance)

                    # Add new particle set to existing tracks
                    self.particle_tracks.append(new_particle_track)

                    # Update ID counter
                    self.id_count += 1                




    def update_particles(self, detections, dt=0.6):
        ### Predict step
        for pf_track in self.particle_tracks:
            pf_track.particles = pf_predict(pf_track.particles, dt, process_noise=self.process_noise)


        ### Update particles track with measurements
        if len(detections) != 0:
            if len(self.particle_tracks) == 0:
                ## Create a new particle set for detection 
                new_particle_set = Particles(self.N)

                # Create gaussian particles from detection center
                # TODO: take first detection for now, in future do hungarian algorithm with detections
                det_x, det_y = detections[0]["center"]  
                new_particle_set.particles = new_particle_set.create_gaussian_particles(mu=(det_x, 0, det_y, 0),
                                                                        std=self.initial_estimate_covariance)

                # Add the particles to the list
                self.particle_tracks.append(new_particle_set)
            else:
                ### Use hungarian assignment to calculate detection particle track pairs
                ## Get the means of each particle set and detection and calculate distance matrix to be used 
                ## in hungarian assignment
                ## TODO: figure out best way computationally to keep track of all the means
                # Extract all track mean position
                all_track_mean_postion = []
                for pf_track in self.particle_tracks:
                    track_mu, track_cov = estimate_particles(pf_track.particles, pf_track.weights)
                    track_position = np.array([track_mu[0], track_mu[2]])
                    all_track_mean_postion.append(track_position)
                all_track_mean_postion = np.array(all_track_mean_postion)

                # Extract all detections and place in matrix
                detection_center_positions = []
                for detection in detections:
                    detection_center = detection["center"]
                    detection_center_positions.append(detection_center)
                detection_center_positions = np.array(detection_center_positions)

                # Calculate euclidean distance matrix with detection centers and particle means
                euclidean_distance_matrix = distance_matrix(all_track_mean_postion, detection_center_positions)

                # Find optimal assignments with hungarian
                row_ind, col_ind = linear_sum_assignment(euclidean_distance_matrix)

                assigned_detections = np.array([detection_center_positions[col_ind[i]] for i in range(len(col_ind))])
                unassigned_detections = np.delete(detection_center_positions, col_ind, axis=0)


                print(detection_center_positions)
                print("Euclidean distance matrix: \n", euclidean_distance_matrix)
                print("Euclidean distance matrix shape: ", euclidean_distance_matrix.shape)
                print("all_track_mean_postion shape: ", all_track_mean_postion.shape)
                print("detections shape: ", detection_center_positions.shape)
                print("assigned_detections shape: ", assigned_detections.shape)
                print("unassigned_detections shape: ", unassigned_detections.shape)

                print("col_ind: \n", col_ind)
                print("------------")


                # print("Assigned Detection Centers: \n", assigned_detections)
                # print("Unassigned Detections: \n", unassigned_detections)



                ### For each assigned detection, update the corresponding particle sets
                ## Particle sets should be in the order with the assigned_detections
                for i in range(len(assigned_detections)):
                    print("all track mean ", i, ": ", all_track_mean_postion[i])
                    print("assigned_detections ", i, ": ", assigned_detections[i])
                    ## Get measurement of tracked object w.r.t to detection
                    dist_from_det_to_mean = np.linalg.norm(all_track_mean_postion[i] - assigned_detections[i])
                    print("Dist from det to mean: ", dist_from_det_to_mean)

                    self.particle_tracks[i].weights = pf_update(self.particle_tracks[i].particles, 
                                                            self.particle_tracks[i].weights, 
                                                            dist_from_det_to_mean, 
                                                            sensor_noise=2, 
                                                            detection_landmark=assigned_detections[i])

                    # Resample if we drop below the number of effective particles
                    if neff(self.particle_tracks[i].weights) < (len(self.particle_tracks[i].particles) / 2):
                        # # Resample with resampling wheel
                        # particles = resampling_wheel(particles, weights)

                        # Resample with staatified resample
                        indexes = stratified_resample(self.particle_tracks[i].weights)

                        self.particle_tracks[i].particles, self.particle_tracks[i].weights = resample_from_index(
                            self.particle_tracks[i].particles, self.particle_tracks[i].weights, indexes)
                        assert np.allclose(self.particle_tracks[i].weights, 1/self.N)


                ### For each unassigned detection, create new particle set
                for i in range(len(unassigned_detections)):
                    ## Create a new particle set for detection 
                    new_particle_set = Particles(self.N)

                    # Create gaussian particles from the unassigned detection center
                    det_x, det_y = unassigned_detections[i]
                    new_particle_set.particles = new_particle_set.create_gaussian_particles(mu=(det_x, 0, det_y, 0),
                                                                            std=self.initial_estimate_covariance)

                    # Add the particles to the list
                    self.particle_tracks.append(new_particle_set)


                # for pf_track in self.particle_tracks:
                #     ## Get measurement of tracked object w.r.t to detection
                #     track_mu, track_cov = estimate_particles(pf_track.particles, pf_track.weights)
                #     track_postion = np.array([track_mu[0], track_mu[2]])

                #     dist_from_det_to_mean = np.linalg.norm(track_postion - detections[0]["center"])
                #     print("Dist from det to mean: ", dist_from_det_to_mean)

                #     pf_track.weights = pf_update(pf_track.particles, pf_track.weights, dist_from_det_to_mean, sensor_noise=2, 
                #                             detection_landmark=detections[0]["center"])


                #     # Resample if we drop below the number of effective particles
                #     if neff(pf_track.weights) < (len(pf_track.particles) / 2):
                #         # # Resample with resampling wheel
                #         # particles = resampling_wheel(particles, weights)

                #         # Resample with staatified resample
                #         indexes = stratified_resample(pf_track.weights)

                #         pf_track.particles, pf_track.weights = resample_from_index(pf_track.particles, pf_track.weights, indexes)
                #         assert np.allclose(pf_track.weights, 1/self.N)



