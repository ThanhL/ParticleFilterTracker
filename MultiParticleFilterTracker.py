import numpy as np
from Particles import Particles
from ParticleFilter import *
from scipy.spatial import distance_matrix
from scipy.optimize import linear_sum_assignment


class MultiParticleFilterTracker():
    def __init__(self, N):
        # Particle set for each unique object
        self.particle_tracks = []

        # Number of particles to use across all sets
        self.N = N

        # Initial estimate covariance
        self.initial_estimate_covariance = np.array([4, 2, 4, 2])

        # Gaussian process noise for states x, x_dot, y, y_dot
        self.process_noise = [0.2,2,0.2,2]


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



