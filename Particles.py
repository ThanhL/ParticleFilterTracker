import numpy as np

class Particles():
    def __init__(self, N):
        self.N = N

        ## Initialize particles with weights
        self.particles = np.empty((N, 4))
        self.weights = np.ones(N) / N


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


if __name__ == "__main__":
    ## Particle Filter params
    N = 400              # Number of particles
    state_dim = 4       # x, x_dot, y, y_dot

    ## Gaussian set of particles
    mu = [5,0,5,0]                 # initial x, x_dot, y, y_dot 
    std = [0.5, 4, 0.5, 4]              # inital stddev of x, x_dot, y, y_dot 


    particle_set_1 = Particles(N=N)
    particle_set_1.particles = particle_set_1.create_gaussian_particles(mu=mu, std=std)


    print("Particles:\n", particle_set_1.particles)
    print("Wegihts:\n", particle_set_1.weights)

    plt.scatter(particle_set_1.particles[:,0],
            particle_set_1.particles[:,2],
            color='k', 
            marker=',', 
            s=1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((-10,10))
    plt.ylim((-10,10))
    plt.grid(True)
    plt.show()