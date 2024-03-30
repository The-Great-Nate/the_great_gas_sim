#include <iostream>
#include <vector>
#include <cmath>
#include "vector3d.cpp"



// Constants
const double box_size = 10.0; // Size of the simulation box
const double epsilon = 1.0; // Lennard-Jones potential parameter
const double sigma = 1.0; // Lennard-Jones potential parameter
const double dt = 0.001; // Time step size
const int N = 2;
const int steps = 100000;

struct Particle{
    vec a, v, r;
    double mass;
};

double lj_pot(vec r1, vec r2){
    vec diff = r2 - r1;
    double r = diff.length();
    double r_term = pow(sigma/r, 12) - pow(sigma    /r, 6);
    return 4 * epsilon * r_term;
};

vec lj_force(vec r1, vec r2){
    vec diff = r2 - r1;
    double r_mag = diff.length();
    vec r_hat = diff/r_mag;
    return ((24.0 * epsilon)/sigma) * (-2.0*(pow(sigma/r_mag, -13))+(pow(sigma/r_mag,-7))) * r_hat;
};

void verlet(std::vector<Particle>& particles){
    int N = particles.size();

    for (int i = 0; i < N; ++i) {
        particles[i].v.set(particles[i].v.x() + 0.5 * dt * particles[i].a.x(), particles[i].v.y() + 0.5 * dt * particles[i].a.y(), particles[i].v.z() + 0.5 * dt * particles[i].a.z()); 
    }

    for (int i = 0; i < N; ++i) {
        particles[i].r.set(particles[i].r.x() + dt * particles[i].v.x(), particles[i].r.y() + dt * particles[i].v.y(), particles[i].r.z() + dt * particles[i].v.z()); 
    }
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j){
            if (i != j) {
                vec force = lj_force(particles[i].r, particles[j].r);
                particles[i].a += force / particles[i].mass;
            }
        }
    }

    for (int i = 0; i < steps; ++i) {
        particles[i].v.set(particles[i].v.x() + 0.5 * dt * particles[i].a.x(), particles[i].v.y() + 0.5 * dt * particles[i].a.y(), particles[i].v.z() + 0.5 * dt * particles[i].a.z()); 
    }
}

