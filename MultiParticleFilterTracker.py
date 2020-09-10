import numpy as np
from Particles import Particles
from ParticleFilter import *

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


    def update_particles(self, detections, dt=0.8):

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
                for pf_track in self.particle_tracks:
                    ## Get measurement of tracked object w.r.t to detection
                    track_mu, track_cov = estimate_particles(pf_track.particles, pf_track.weights)
                    track_postion = np.array([track_mu[0], track_mu[2]])

                    dist_from_det_to_mean = np.linalg.norm(track_postion - detections[0]["center"])
                    print("Dist from det to mean: ", dist_from_det_to_mean)

                    pf_track.weights = pf_update(pf_track.particles, pf_track.weights, dist_from_det_to_mean, sensor_noise=2, 
                                            detection_landmark=detections[0]["center"])


                    # Resample if we drop below the number of effective particles
                    if neff(pf_track.weights) < (len(pf_track.particles) / 2):
                        # # Resample with resampling wheel
                        # particles = resampling_wheel(particles, weights)

                        # Resample with staatified resample
                        indexes = stratified_resample(pf_track.weights)

                        pf_track.particles, pf_track.weights = resample_from_index(pf_track.particles, pf_track.weights, indexes)
                        assert np.allclose(pf_track.weights, 1/self.N)


