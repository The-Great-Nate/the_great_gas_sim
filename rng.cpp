#include <iostream>
#include <random>
#include <chrono>

struct RandomNumbers
{
    std::vector<double> rand_x, rand_y, rand_z;
    RandomNumbers(int N) { rand_x.resize(N); rand_y.resize(N); rand_z.resize(N); };
};

RandomNumbers generate(int N, float box_size)
{
    const double half_box = box_size / 2.0;
    // Initialise the random number generator:
    RandomNumbers random_nums(N);
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distribution(-half_box, half_box);

    // Initialise a clock object
    typedef std::chrono::high_resolution_clock myclock;
    myclock::time_point beginning = myclock::now();

    // Obtain a seed from the timer and apply it
    myclock::duration d = myclock::now() - beginning;
    unsigned seed = d.count();
    generator.seed(seed); // Apply the seed

    for (int i = 0; i < N; i++)
    {
        random_nums.rand_x[i] = distribution(generator);
        random_nums.rand_y[i] = distribution(generator);
        random_nums.rand_z[i] = distribution(generator);
    }
    return random_nums;
}