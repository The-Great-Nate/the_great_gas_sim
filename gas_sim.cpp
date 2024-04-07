#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <string>
#include <filesystem>
#include <iomanip>
#include "vector3d.cpp"
#include "rng.cpp"

#define K_B 1.380649E-23
#define u 1.660538921E-24

// Constants
const double epsilon = 125.7 * K_B;                     // Lennard-Jones potential parameter
const double sigma = 0.3345E-9;                         // Lennard-Jones potential parameter
const double ma = 39.948 * u;                           // Reference mass (particle mass)
const double t0 = sqrt((ma * pow(sigma, 2)) / epsilon); // Reference time
const double box_size = 2;
const double dimentionless_mass = ma / ma;
const double half_box = box_size / 2.0;
int N;
int steps;
double dt; // Time step size

struct Particle
{
    vec a, v, r;
    double mass;
    double PE = 0;
    Particle(double mass_) { mass = mass_; };
};

vec calculate_kinetic_energy(const Particle particle)
{
    vec E_k = vec(0.5 * particle.mass, 0.5 * particle.mass, 0.5 * particle.mass) * particle.v.pow(2);
    return E_k;
}

double lj_pot(vec r1, vec r2)
{
    vec diff = r2 - r1;
    double r = diff.length();
    double r_term = pow(sigma / r, 12) - pow(sigma / r, 6);
    return 4 * epsilon * r_term;
};

vec lj_force(vec r1, vec r2)
{
    vec diff = r2 - r1;
    double r_mag = diff.length();
    vec r_hat = diff / r_mag;
    return ((24.0 * epsilon) / sigma) * (-2.0 * (pow(sigma / r_mag, 13)) + (pow(sigma / r_mag, 7))) * r_hat;
};

double dimensionless_lj_pot(vec r1, vec r2)
{
    vec diff = r2 - r1;
    double r = diff.length();
    double r_term = pow(1 / r, 12) - pow(1 / r, 6);
    return 4.0 * r_term;
};

vec dimensionless_lj_force(vec r1, vec r2)
{
    vec diff = r2 - r1;
    double r_mag = diff.length();
    vec r_hat = diff / r_mag;
    return ((24.0 * 1.0) / 1.0) * (-2.0 * (pow(1.0 / r_mag, 13)) + (pow(1.0 / r_mag, 7))) * r_hat;
};

void write_data(const Particle particle, int n, int step, std::ofstream &gas_file)
{
    if (gas_file.is_open())
    {
        vec KE = calculate_kinetic_energy(particle);
        gas_file << dt * step << "\t" << n << "\t"
                 << particle.r.x() << "\t" << particle.r.y() << "\t" << particle.r.z() << "\t"
                 << particle.v.x() << "\t" << particle.v.y() << "\t" << particle.v.z() << "\t"
                 << particle.a.x() << "\t" << particle.a.y() << "\t" << particle.a.z() << "\t"
                 << KE.x() << "\t" << KE.y() << "\t" << KE.z() << "\t" << particle.PE << "\n";
    }
}

void write_columns(std::ofstream &gas_file)
{
    if (gas_file.is_open())
    {
        gas_file << "t\tn\tx\ty\tz\tvx\tvy\tvz\tax\tay\taz\tKEx\tKEy\tKEz\tPE\n";
    }
}

void check_periodic_conditions(Particle &particle)
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

void verlet(std::vector<Particle> &particles, int step, std::string file_name, std::ofstream &gas_file)
{
    for (int i = 0; i < N; ++i)
    {
        write_data(particles[i], i, step, gas_file);
        particles[i].v.set(particles[i].v.x() + 0.5 * dt * particles[i].a.x(), particles[i].v.y() + 0.5 * dt * particles[i].a.y(), particles[i].v.z() + 0.5 * dt * particles[i].a.z());
        particles[i].r.set(particles[i].r.x() + dt * particles[i].v.x(), particles[i].r.y() + dt * particles[i].v.y(), particles[i].r.z() + dt * particles[i].v.z());
        check_periodic_conditions(particles[i]);
    }

    for (int i = 0; i < N; ++i)
    {
        particles[i].a.set(0, 0, 0);
        particles[i].PE = 0;
        for (int j = 0; j < N; ++j)
        {
            if (i != j)
            {

                vec force = dimensionless_lj_force(particles[i].r, particles[j].r);
                particles[i].PE += dimensionless_lj_pot(particles[i].r, particles[j].r);
                particles[i].a += force;
            }
        }
    }

    for (int i = 0; i < N; ++i)
    {
        particles[i].v.set(particles[i].v.x() + 0.5 * dt * particles[i].a.x(), particles[i].v.y() + 0.5 * dt * particles[i].a.y(), particles[i].v.z() + 0.5 * dt * particles[i].a.z());
    }
}

void random_init_conditions(std::vector<Particle> &particles, std::ofstream &gas_file)
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

void lattice_init_conditions(std::vector<Particle> &particles, std::ofstream &gas_file)
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

int main()
{
    std::string file_name = "";
    std::cout << "What would you like to name this file?: ";
    std::cin >> file_name;
    std::ofstream gas_file;
    std::string data_folder = "data";
    gas_file.open(data_folder + "\\" + file_name + ".txt");

    write_columns(gas_file);

    std::vector<Particle> particles;
    int check_bound;
    std::cout << "Would you like to check if 2 particles are bound to one another? (0 if no, 1 if yes): ";
    std::cin >> check_bound;
    if (check_bound == 1)
    {
        std::cout << "Will set initial coniditons accordingly.\nn is now = 2" << std::endl;
        N = 2;
        particles.resize(N, Particle(ma)); // Resize particles vector
        particles[0].r.set(0.5, 0, 0);
        particles[1].r.set(-0.5, 0, 0);
    }

    else
    {
        std::cout << "How many particles do you want?: ";
        std::cin >> N;

        particles.resize(N, Particle(ma));
        std::string init_cond = "";
        std::cout << "What Init Conditions would you Like?\n- Random\n- Lattice\n ";
        std::cin >> init_cond;
        if (init_cond == "Random")
        {
            random_init_conditions(particles, gas_file);
            std::cout << "Set Random Initial Conditions to all particles!" << std::endl;
        }

        else if (init_cond == "Lattice")
        {
            lattice_init_conditions(particles, gas_file);
            std::cout << "Set Lattice Initial Conditions to all particles!" << std::endl;
        }
    }

    std::cout << "How many timesteps do you want?: ";
    std::cin >> steps;
    std::cout << "What dt do you want?: ";
    std::cin >> dt;

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

    for (int i = 0; i < steps; ++i)
    {
        verlet(particles, i, file_name, gas_file);
    }

    gas_file.close();
    return EXIT_SUCCESS;
}