#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "vector3d.cpp"
#define K_B 1.380649E-23
#define u 1.660538921E-24

// Constants
const double epsilon = 125.7 * K_B; // Lennard-Jones potential parameter
const double sigma = 0.3345E-9;     // Lennard-Jones potential parameter
const int N = 2;
const int steps = 10000;
const double ma = 39.948 * u;                           // Reference mass (particle mass)
const double t0 = sqrt((ma * pow(sigma, 2)) / epsilon); // Reference time
const double dt = 0.01;                                 // Time step size
const double box_size = sigma * 2 / sigma;
const double dimentionless_mass = ma / ma;

struct Particle
{
    vec a, v, r;
    double mass;
    Particle(double mass_) { mass = mass_; };
};

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

vec dimensionless_lj(vec r1, vec r2)
{
    vec diff = r2 - r1;
    double r_mag = diff.length() / sigma;
    vec r_hat = diff / r_mag;
    //std::cout << ((24.0 * 1.0) / 1.0) * (-2.0 * (pow(1.0 / r_mag, 13)) + (pow(1.0 / r_mag, 7))) * r_hat << std::endl;
    return ((24.0 * 1.0) / 1.0) * (-2.0 * (pow(1.0 / r_mag, 13)) + (pow(1.0 / r_mag, 7))) * r_hat;
};

void write_data(const Particle particle, int n, int step)
{
    std::ofstream gas_file;
    gas_file.open("gas_" + std::to_string(n) + ".txt", std::ios_base::app);
    // gas_file << "x\ty\tz\tvx\tvy\tvz\tax\tay\taz\n";
    if (gas_file.is_open())
    {
        gas_file << dt*step << "\t" 
                 << particle.r.x() << "\t" << particle.r.y() << "\t" << particle.r.z() << "\t"
                 << particle.v.x() << "\t" << particle.v.y() << "\t" << particle.v.z() << "\t"
                 << particle.a.x() << "\t" << particle.a.y() << "\t" << particle.a.z() << "\n";
    }
    gas_file.close();
}

void write_columns(int n)
{
    std::ofstream gas_file;
    gas_file.open("gas_" + std::to_string(n) + ".txt");
    // gas_file << "x\ty\tz\tvx\tvy\tvz\tax\tay\taz\n";
    if (gas_file.is_open())
    {
        gas_file << "t\tx\ty\tz\tvx\tvy\tvz\tax\tay\taz\n";
    }
    gas_file.close();
}

void check_periodic_conditions(Particle &particle)
{
    const double half_box = box_size / 2.0;
    if (particle.r.x() > half_box)
    {
        particle.r.set(half_box, particle.r.y(), particle.r.z());
        particle.v.set(-particle.v.x(), particle.v.y(), particle.v.z());
        // std::cout << "Out of Bound Positively X" << std::endl;
    }
    else if (particle.r.x() < -half_box)
    {
        particle.r.set(-half_box, particle.r.y(), particle.r.z());
        particle.v.set(-particle.v.x(), particle.v.y(), particle.v.z());
        // std::cout << "Out of Bound Negatively X" << std::endl;
    }

    if (particle.r.y() > half_box)
    {
        particle.r.set(particle.r.x(), half_box, particle.r.z());
        particle.v.set(particle.v.x(), -particle.v.y(), particle.v.z());
        // std::cout << "Out of Bound Positively Y" << std::endl;
    }
    else if (particle.r.y() < -half_box)
    {
        particle.r.set(particle.r.x(), -half_box, particle.r.z());
        particle.v.set(particle.v.x(), -particle.v.y(), particle.v.z());
        // std::cout << "Out of Bound Negatively Y" << std::endl;
    }

    if (particle.r.z() > half_box)
    {
        particle.r.set(particle.r.x(), particle.r.y(), half_box);
        particle.v.set(particle.v.x(), particle.v.y(), -particle.v.z());
        // std::cout << "Out of Bound Positively Z" << std::endl;
    }
    else if (particle.r.z() < -half_box)
    {
        particle.r.set(particle.r.x(), particle.r.y(), -half_box);
        particle.v.set(particle.v.x(), particle.v.y(), -particle.v.z());
        // std::cout << "Out of Bound Negavitvely Z" << std::endl;
    }
}

void verlet(std::vector<Particle> &particles, int step)
{
    int N = particles.size();

    for (int i = 0; i < N; ++i)
    {
        write_data(particles[i], i, step);
    }

/* This force loop was performed twice in initial testing of the simulation. Will be deleted
    for (int i = 0; i < N; ++i)
    {
        // particles[i].a.set(0, 0, 0);
        for (int j = 0; j < N; ++j)
        {
            // std::cout << "a " << particles[i].a << i << std::endl;
            if (i != j)
            {
                vec force = dimensionless_lj(particles[i].r, particles[j].r);
                particles[i].a += force;
            }
        }
    }
*/

    for (int i = 0; i < N; ++i)
    {

        // std::cout << "v " << particles[i].v << i << std::endl;
        particles[i].v.set(particles[i].v.x() + 0.5 * dt * particles[i].a.x(), particles[i].v.y() + 0.5 * dt * particles[i].a.y(), particles[i].v.z() + 0.5 * dt * particles[i].a.z());
    }

    for (int i = 0; i < N; ++i)
    {
        // std::cout << "r " << particles[i].r << i << std::endl;
        particles[i].r.set(particles[i].r.x() + dt * particles[i].v.x(), particles[i].r.y() + dt * particles[i].v.y(), particles[i].r.z() + dt * particles[i].v.z());
        check_periodic_conditions(particles[i]);
    }

    for (int i = 0; i < N; ++i)
    {
        // particles[i].a.set(0, 0, 0);
        for (int j = 0; j < N; ++j)
        {
            // std::cout << "a " << particles[i].a << i << std::endl;
            if (i != j)
            {
                vec force = dimensionless_lj(particles[i].r, particles[j].r);
                particles[i].a += force;
            }
        }
    }

    for (int i = 0; i < N; ++i)
    {
        particles[i].v.set(particles[i].v.x() + 0.5 * dt * particles[i].a.x(), particles[i].v.y() + 0.5 * dt * particles[i].a.y(), particles[i].v.z() + 0.5 * dt * particles[i].a.z());
    }
}

int main()
{
    std::vector<Particle> particles(N, Particle(ma));
    particles[0].r.set(0.99-sigma, 0, 0);
    particles[1].r.set(0.99, 0, 0);
    for (int i = 0; i < N; ++i)
    {
        write_columns(i);
    }
    for (int i = 0; i < steps; ++i)
    {
        verlet(particles, i);
    }
    return EXIT_SUCCESS;
}
