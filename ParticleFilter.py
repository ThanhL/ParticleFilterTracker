import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from Particles import Particles


## Resampling
# Stratified Resampling strategy for particles
# Adapted from rlabbe: https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/12-Particle-Filters.ipynb
# Stratified resample
def stratified_resample(particles, weights):
    N = len(weights)

    ### Determine indexes using stratified strategy
    positions = (np.random.random(N) + range(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1

    ### Resample the particles with the indexes from stratified strategy
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))

    return particles

# Resampling Wheel
# From AI for Robotics Sebastian Thrun Course
def resampling_wheel(particles,weights):
    N = len(particles)

    resampled_particles = []

    # Generate indexes
    index = np.random.randint(0, N, 1).squeeze()
    # Initialize beta
    beta = 0.0
    # Initialize max weight
    max_weight = np.max(weights)

    for i in range(N):
        beta = beta + np.random.uniform(0, 2*max_weight, 1).squeeze()

        while beta > weights[index]:
            beta = beta - weights[index]
            index = (index + 1) % N  # Mod for circular motion around array
        resampled_particles.append(particles[index])

    # Reinit weights
    weights.fill(1.0 / len(weights))

    return np.array(resampled_particles)


## Number of effective weights
def neff(weights):
    return 1. / np.sum(np.square(weights))

## Particle filter localization step (Table 4.3 Probablistic Robots)
def particle_filter(particles, weights, u_t, z_t):
    # Predict!
    particles = pf_predict(particles, u_t, DT)

    # Update!
    weights = pf_update(particles, weights, z_t)


    # Resample if we drop below the number of effective particles
    if neff(weights) < (len(particles) / 2):
        # # Resample with resampling wheel
        # particles = resampling_wheel(particles, weights)

        # Resample with staatified resample
        indexes = stratified_resample(weights)
        resample_from_index(particles, weights, indexes)

        assert np.allclose(weights, 1/N)

    return particles, weights

# Particle filter prediction step with bicycle model
def pf_predict(particles, dt, process_noise):
    """
    Use basic newtonian equation of motion (velocity + position) model to update particles. Since no input
    particles are predicted using their velocity as the input.
    

    particles: particles with each particle representing state [x, x_dot, y, y_dot].T

    dt: delta time
    process_noise: Process noise for each state [x, x_dot, y, y_dot].T
    """
    N = len(particles)

    # Update x and x_dot
    particles[:,0] += particles[:,1] * dt +  (np.random.randn(N) * process_noise[0])
    particles[:,1] += (np.random.randn(N) * process_noise[1])
    # Update y and y_dot
    particles[:,2] += particles[:,3] * dt + (np.random.randn(N) * process_noise[2])
    particles[:,3] += (np.random.randn(N) * process_noise[3])
    return particles


# Particle filter update step with landmarks and measurements
def pf_update(particles, weights, z_t, sensor_noise, detection_landmark):
    # Iterate through each landmark and calculate the weighting probability
    # Calculate weighting probability with respect to measurement from detector
    # Detection gives x,y position, so we will use the euclidean dist
    # from each particle w.r.t to dist to calculate the weighting
    # w_t[m] = p(z_t | x_t[m])
    N = len(particles)

    position_particles = np.hstack((particles[:, 0].reshape(-1,1), particles[:, 2].reshape(-1,1)))
    dist = np.linalg.norm(position_particles - detection_landmark, axis=1)
    weights = scipy.stats.norm(dist, sensor_noise).pdf(z_t)

    weights += 1.e-300              # For round off error
    weights /= np.sum(weights)      # Normalize weights 
    return weights

# Estimated state from particles by comput weighted mean and covariance of particle distribution
def estimate_particles(particles, weights):
    mean = np.average(particles, weights=weights, axis=0)
    covariance_matrix = np.average((particles - mean)**2, weights=weights, axis=0)
    return mean, covariance_matrix 


#### Testing Particle Filters
N = 100
DT = 0.1
SIM_TIME = 20.0
SHOW_ANIMATION = True

# Plotting params
xlim = (-20,20)
ylim = (-20,20)


if __name__ == "__main__":
    ## Starting time
    time = 0.0

    ## Particles
    test_particles = Particles(N=N)
    test_particles.particles = test_particles.create_gaussian_particles(mu=[5,0,5,0], std=[0.2, 2, 0.2, 2])
    

    ## Run the simulation 
    while SIM_TIME >= time:
        # Update sim time
        time += DT

        # Run particles into prediction
        test_particles.particles = pf_predict(test_particles.particles, DT,
                                                    process_noise=[0.2,0.2,0.2,0.2])



        ### Plot particles when piped into particle filter
        if SHOW_ANIMATION:
            plt.cla()

            plt.scatter(test_particles.particles[:,0],
                    test_particles.particles[:,2],
                    color='k', 
                    marker=',', 
                    s=1)


            plt.xlabel('x')
            plt.ylabel('y')
            plt.xlim(*xlim)
            plt.ylim(*ylim)
            plt.grid(True)
            plt.pause(0.1)