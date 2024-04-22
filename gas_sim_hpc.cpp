#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <filesystem>
#include <iomanip>
#include <chrono>
#include <sstream>
#include "vector3d.cpp"
#include "rng.cpp"

// Universal Physical Constants
#define K_B 1.380649E-23
#define u 1.660538921E-24

// Constants
const double epsilon = 125.7 * K_B;                     // Lennard-Jones potential parameter
const double sigma = 0.3345E-9;                         // Lennard-Jones potential parameter
const double ma = 39.948 * u;                           // Reference mass (particle mass)
const double t0 = sqrt((ma * pow(sigma, 2)) / epsilon); // Reference time
const double dimentionless_mass = ma / ma;              // Dimentionless mass

// Initialise variables
std::string init_conditions;
int N; // No. of Particles
int duration;
int steps;
double dt; // Time step size
double box_size;
/**
 * Structure of a particle.
 *
 * Made of vecs of a: acceleration, v: velocity, r: position.
 * Mass and energy wrt to other particles held in variables.
 *
 * @param[in] Mass of the particle.
 * */
struct Particle
{
    vec a, v, r;
    double mass;
    double PE = 0;
    Particle(double mass_) { mass = mass_; };
};


/**
 * Calculates potential energy.
 *
 * Uses the Lennard-Jones potential
 * V(r) = 4 * epsilon * [(sigma/r)^12 - (sigma/r)^6].
 * where epsilon and sigma are constants defined above.
 * r = seperation distance between particles.
 *
 * @param[in] r1, r2 vecs which hold the position of the particles.
 * @return Lennard-Jones potential energy between particles in r1 and r2.
 * */
double lj_pot(vec r1, vec r2)
{
    vec diff = r2 - r1;
    double r = diff.length();
    double r_term = pow(sigma / r, 12) - pow(sigma / r, 6);
    return 4 * epsilon * r_term;
};

/**
 * Calculates force between particles.
 *
 * Uses the Lennard-Jones potential and the fact that force = negative gradient of potential energy.
 * Uses analytical solution to find force.
 * F_12 = 24 * (epsilon/sigma) * [-2*(sigma/r)^13 + (sigma/r)^7] * r_hat.
 * where epsilon and sigma are constants defined above.
 * r_hat is the unit vector between the two particles
 * r = seperation distance between particles.
 *
 * @param[in] r1, r2 vecs which hold the position of the particles
 * @return Force between particles in r1 and r2.
 * */
vec lj_force(vec r1, vec r2)
{
    vec diff = r2 - r1;
    double r_mag = diff.length();
    vec r_hat = diff / r_mag;
    return ((24.0 * epsilon) / sigma) * (-2.0 * (pow(sigma / r_mag, 13)) + (pow(sigma / r_mag, 7))) * r_hat;
};

/**
 * Calculates potential energy in dimentionless formulation.
 *
 * Uses the dimentionless formulation form of the Lennard-Jones potential
 * V(r) = 4 * [(1/r)^12 - (1/r)^6].
 * r = seperation distance between particles.
 *
 * @param[in] r1, r2 vecs which hold the position of the particles.
 * @return Lennard-Jones potential energy between particles in r1 and r2.
 * */
double dimensionless_lj_pot(vec r1, vec r2)
{
    vec diff = r2 - r1;
    double r = diff.length();
    double r_term = pow(1 / r, 12) - pow(1 / r, 6);
    return 4.0 * r_term;
};

/**
 * Calculates force between particles using dimentionless formulation.
 *
 * Uses the Lennard-Jones potential and the fact that force = negative gradient of potential energy.
 * Uses analytical solution to find force.
 * F_12 = 24 * [-2*(1/r)^13 + (1/r)^7] * r_hat.
 * where epsilon and sigma are constants defined above.
 * r_hat is the unit vector between the two particles
 * r = seperation distance between particles.
 *
 * @param[in] r1, r2 vecs which hold the position of the particles.
 * @return Force between particles in r1 and r2.
 * */
vec dimensionless_lj_force(vec r1, vec r2)
{
    vec diff = r2 - r1;
    double r_mag = diff.length();
    vec r_hat = diff / r_mag;
    return 24.0 * (-2.0 * (pow(1.0 / r_mag, 13)) + (pow(1.0 / r_mag, 7))) * r_hat;
};

/**
 * Writes attributes of particle into text file.
 *
 * Seperates each attribute with a tab.
 *
 * @param[in] particle, n, step input data to write into file.
 * @param[in] gas_file file object for data to be written to.
 * */
void write_data(const Particle particle, int n, int step, std::ofstream &gas_file)
{
    if (gas_file.is_open())
    {
        gas_file << dt * step << "\t" << n << "\t"
                 << particle.r.x() << "\t" << particle.r.y() << "\t" << particle.r.z() << "\t"
                 << particle.v.x() << "\t" << particle.v.y() << "\t" << particle.v.z() << "\t"
                 << particle.a.x() << "\t" << particle.a.y() << "\t" << particle.a.z() << "\t"
                 << particle.PE << "\n";
    }
}

/**
 * Writes system parameters into text file.
 * "table" is written to indicate the start of the dataset.
 * Main purpose to serve the extract_parameters() function in the data_analysis_package.
 *
 * @param[in] gas_file file object for data to be written to.
 * */
void write_params(std::ofstream &gas_file)
{
    if (gas_file.is_open())
    {
        gas_file << "N=" << N << "\n"
                 << "box_size=" << box_size << "\n"
                 << "steps=" << steps << "\n"
                 << "dt=" << dt << "\n"
                 << "init_conditions=" << init_conditions << "\n"
                 << "table"
                 << "\n";
    }
}

/**
 * Writes columns into text file.

 * @param[in] gas_file file object for data to be written to.
 * */
void write_columns(std::ofstream &gas_file)
{
    if (gas_file.is_open())
    {
        gas_file << "t\tn\tx\ty\tz\tvx\tvy\tvz\tax\tay\taz\tKEx\tKEy\tKEz\tPE\n";
    }
}

/**
 * Check if particle's position update is out of bounds of box
 *
 * If it is, put back in box and flip velocity direction for whichever axis this applies to
 *
 * @param[in] particle particle in question
 * @param[in] half_box maximum co ordinate (positive and negative) particle can be at
 * */
void check_periodic_conditions(Particle &particle, const double half_box)
{
    if (particle.r.x() > half_box)
    {
        particle.r.set(half_box, particle.r.y(), particle.r.z());
        particle.v.set(-particle.v.x(), particle.v.y(), particle.v.z());
    }
    else if (particle.r.x() < -half_box)
    {
        particle.r.set(-half_box, particle.r.y(), particle.r.z());
        particle.v.set(-particle.v.x(), particle.v.y(), particle.v.z());
    }

    if (particle.r.y() > half_box)
    {
        particle.r.set(particle.r.x(), half_box, particle.r.z());
        particle.v.set(particle.v.x(), -particle.v.y(), particle.v.z());
    }
    else if (particle.r.y() < -half_box)
    {
        particle.r.set(particle.r.x(), -half_box, particle.r.z());
        particle.v.set(particle.v.x(), -particle.v.y(), particle.v.z());
    }

    if (particle.r.z() > half_box)
    {
        particle.r.set(particle.r.x(), particle.r.y(), half_box);
        particle.v.set(particle.v.x(), particle.v.y(), -particle.v.z());
    }
    else if (particle.r.z() < -half_box)
    {
        particle.r.set(particle.r.x(), particle.r.y(), -half_box);
        particle.v.set(particle.v.x(), particle.v.y(), -particle.v.z());
    }
}

/**
 * Performs a timestep of the Verlet 'leap-frog' integration scheme
 *
 * If it is, put back in box and flip velocity direction for whichever axis this applies to
 *
 * @param[in] particles vector of particles.
 * @param[in] step current step taking place.
 * @param[in] half_box maximum co ordinate (positive and negative) particle can be at.
 * @param[in] gas_file file object for data to be written to.
 * */
void verlet(std::vector<Particle> &particles, int step, const double half_box, std::ofstream &gas_file)
{
    // Perform write_data(), velocity and position updates on each particle.
    for (int i = 0; i < N; ++i)
    {
        write_data(particles[i], i, step, gas_file);
        particles[i].v.set(particles[i].v.x() + 0.5 * dt * particles[i].a.x(), particles[i].v.y() + 0.5 * dt * particles[i].a.y(), particles[i].v.z() + 0.5 * dt * particles[i].a.z());
        particles[i].r.set(particles[i].r.x() + dt * particles[i].v.x(), particles[i].r.y() + dt * particles[i].v.y(), particles[i].r.z() + dt * particles[i].v.z());
        check_periodic_conditions(particles[i], half_box);
    }

    // Update acceleration through summing forces from each particle
    for (int i = 0; i < N; ++i)
    {
        // Reset acceleration and PE on a particle after each timestep
        particles[i].a.set(0, 0, 0);
        particles[i].PE = 0;
        for (int j = 0; j < N; ++j)
        {
            if (i != j)
            {
                // Run force and potential calculations.
                vec force = dimensionless_lj_force(particles[i].r, particles[j].r);
                particles[i].PE += dimensionless_lj_pot(particles[i].r, particles[j].r);
                particles[i].a += force;
            }
        }
    }

    // Perform final velocity update on each particle.
    for (int i = 0; i < N; ++i)
    {
        particles[i].v.set(particles[i].v.x() + 0.5 * dt * particles[i].a.x(), particles[i].v.y() + 0.5 * dt * particles[i].a.y(), particles[i].v.z() + 0.5 * dt * particles[i].a.z());
    }
}

/**
 * Places particles in a random position within the box
 *
 * If it is, put back in box and flip velocity direction for whichever axis this applies to
 *
 * @param[in] particles vector of particles.
 * */
void random_init_conditions(std::vector<Particle> &particles)
{
    RandomNumbers rand_nums(N);
    rand_nums = generate(N, box_size);
    std::vector<double> rand_x = rand_nums.rand_x;
    std::vector<double> rand_y = rand_nums.rand_y;
    std::vector<double> rand_z = rand_nums.rand_z;

    for (int i = 0; i < N; ++i)
    {
        particles[i].r.set(rand_x[i], rand_y[i], rand_z[i]);
    }
}

/**
 * Places particles in a box as close to each other as possible before instability occurs
 *
 * @param[in] particles vector of particles.
 * */
void lattice_init_conditions(std::vector<Particle> &particles)
{
    // Calculate the number of particles along each axis
    int num_per_axis = ceil(pow(N, 1.0 / 3.0));

    // Initialize particles with evenly spaced positions around the origin
    int index = 0;
    for (int i = 0; i < num_per_axis; ++i)
    {
        for (int j = 0; j < num_per_axis; ++j)
        {
            for (int k = 0; k < num_per_axis; ++k)
            {
                if (index >= N)
                    break; // Prevent out-of-bounds access

                // Calculate the position of the particle along each axis
                double x = (i - num_per_axis / 2.0) * 1;
                double y = (j - num_per_axis / 2.0) * 1;
                double z = (k - num_per_axis / 2.0) * 1;

                particles[index].r.set(x, y, z);
                index += 1;
            }
        }
    }
}

/**
 * main() asks user for system parameters and types of initial conditions desired.
 *
 * Upon completion, a txt file in the data folder is generated.
 *
 * @return EXIT_SUCCESS :) :) :)
 * */
int main(int argc, char **argv)
{
    // Checking if number of argument is equal to 4 or not.
    if (argc < 5 || argc > 6)
    {
        cout << "ERROR: need 5 input arguments - gas_sim N BX dt duration init_conditions" << endl;
        return EXIT_FAILURE;
    }

    N = atoi(argv[1]);
    box_size = atoi(argv[2]);
    dt = atof(argv[3]);
    duration = atoi(argv[4]);
    std::string init_cond = argv[5];
    const double half_box = box_size / 2.0;

    // Initialise Particle vector
    std::vector<Particle> particles;
    particles.resize(N, Particle(ma)); // Set particles vector to size N with mass ma

    // Applys init cond
    if (init_cond == "Random")
    {
        init_conditions = "random";
        random_init_conditions(particles);
        std::cout << "Set Random Initial Conditions to all particles!" << std::endl;
    }

    else if (init_cond == "Lattice")
    {
        init_conditions = "lattice";
        lattice_init_conditions(particles);
        std::cout << "Set Lattice Initial Conditions to all particles!" << std::endl;
    }

    steps = duration / dt;

    /* Creates ofstream object and open txt file of file_name in data folder.
    Makes filename from parameters of system.
    */
    std::ofstream gas_file;
    std::string data_folder = "data";

    /* Truncates trailing 0's from double variables to be written as filename
        the .str() later retrieves string from both ostringstream objects
        .erase() removes characters from a string
        .find_last_not_of() finds first character that doesnt match arguement in reverse order
        std::string::npos entire string is searched.
    */
    std::string box_size_trunc = std::to_string(box_size);
    std::string dt_trunc = std::to_string(dt);
    box_size_trunc.erase(box_size_trunc.find_last_not_of('0') + 1, std::string::npos);
    box_size_trunc.erase(box_size_trunc.find_last_not_of('.') + 1, std::string::npos);
    dt_trunc.erase(dt_trunc.find_last_not_of('0') + 1, std::string::npos);
    dt_trunc.erase(dt_trunc.find_last_not_of('.') + 1, std::string::npos);

    // Making File based of system params
    std::string file_name = "N_" + std::to_string(N) + "__" +
                            "BX_" + box_size_trunc + "__" +
                            "dt_" + dt_trunc + "__" +
                            "duration_" + std::to_string(duration) + "__" +
                            "init_cond_" + init_conditions;
    gas_file.open(data_folder + "\\" + file_name + ".txt");

    // Start the clock for how long this simulation takes to run
    auto start = std::chrono::high_resolution_clock::now();
    write_params(gas_file);
    write_columns(gas_file);

    // Calculate initial potential energies of each particle before vertlet loop begins
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            if (i != j)
            {
                particles[i].PE += dimensionless_lj_pot(particles[i].r, particles[j].r);
            }
        }
    }

    // Vertlet loop begins!
    for (int i = 0; i < steps; ++i)
    {
        verlet(particles, i, half_box, gas_file);
    }

    // Stop the clock for how long this simulation takes to run
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    auto minutes = std::chrono::duration_cast<std::chrono::minutes>(duration);
    auto seconds = duration - minutes;
    std::cout << "Simulation Runtime " << minutes.count() << ":" << seconds.count() << " (MM:ss) \n";
    gas_file << "duration=" << minutes.count() << ":" << seconds.count();

    // close gas file.
    gas_file.close();

    return EXIT_SUCCESS;
}