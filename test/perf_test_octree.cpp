#include <gtest/gtest.h>
#include <taskflow/taskflow.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "octree/Octree.h"
#include "util/Hilbert.h"

using KeyType = std::uint64_t;

namespace
{

std::vector<float> generateRandomCoordinates(size_t numPoints, float min = -100.0f, float max = 100.0f)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min, max);

    std::vector<float> coords(numPoints * 3);
    for (size_t i = 0; i < coords.size(); ++i)
    {
        coords[i] = dis(gen);
    }
    return coords;
}

void splitCoordinates(const std::vector<float> &coords,
                      std::vector<float> &x,
                      std::vector<float> &y,
                      std::vector<float> &z)
{
    size_t numPoints = coords.size() / 3;
    x.resize(numPoints);
    y.resize(numPoints);
    z.resize(numPoints);

    for (size_t i = 0; i < numPoints; ++i)
    {
        x[i] = coords[i * 3];
        y[i] = coords[i * 3 + 1];
        z[i] = coords[i * 3 + 2];
    }
}

} // namespace

TEST(OctreePerformance, BuildTime)
{
    std::vector<size_t> pointCounts = {1000, 10000, 100000, 1000000};

    std::vector<double> buildTimes;
    tf::Executor executor(1);

    std::cout << "\n=== Octree Build Performance Test ===\n";
    std::cout << "| #Points | Build Time (ms) | Time per Point (ns) |\n";
    std::cout << "|---------|-----------------|---------------------|\n";

    for (size_t numPoints : pointCounts)
    {
        auto coords = generateRandomCoordinates(numPoints);

        bvh2::Box<float> box;
        for (size_t i = 0; i < numPoints; ++i)
        {
            bvh2::Vec3<float> point(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
            box.expand(point);
        }

        for (int i = 0; i < 3; ++i)
        {
            float range = box.max[i] - box.min[i];
            box.min[i] -= range * 0.01f;
            box.max[i] += range * 0.01f;
        }

        std::vector<float> x, y, z;
        splitCoordinates(coords, x, y, z);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<KeyType> codes(numPoints);
        bvh2::computeSfcKeys(x.data(), y.data(), z.data(), codes.data(), numPoints, box, executor);
        auto codeEnd = std::chrono::high_resolution_clock::now();

        std::sort(codes.begin(), codes.end());
        auto sortEnd = std::chrono::high_resolution_clock::now();

        unsigned bucketSize = 16;
        cstone::Octree<KeyType> octree(bucketSize);
        auto buildStart = std::chrono::high_resolution_clock::now();
        octree.build(codes.data(), codes.data() + codes.size(), executor);
        auto buildEnd = std::chrono::high_resolution_clock::now();

        auto codeTime = std::chrono::duration<double, std::milli>(codeEnd - start).count();
        auto sortTime = std::chrono::duration<double, std::milli>(sortEnd - codeEnd).count();
        auto octreeTime = std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();
        auto totalTime = std::chrono::duration<double, std::milli>(buildEnd - start).count();

        double timePerPoint = (totalTime * 1e6) / numPoints;
        buildTimes.push_back(totalTime);

        std::cout << "| " << std::setw(7) << numPoints << " | "
                  << std::setw(15) << std::fixed << std::setprecision(2) << totalTime << " | "
                  << std::setw(19) << std::fixed << std::setprecision(2) << timePerPoint << " |\n";

        std::cout << "  - Key generation: " << std::fixed << std::setprecision(2) << codeTime << " ms\n";
        std::cout << "  - Key sorting:    " << std::fixed << std::setprecision(2) << sortTime << " ms\n";
        std::cout << "  - Octree build:   " << std::fixed << std::setprecision(2) << octreeTime << " ms\n\n";

        const auto &counts = octree.counts();
        ASSERT_EQ(std::accumulate(counts.begin(), counts.end(), size_t(0)), numPoints);
    }

    EXPECT_FALSE(buildTimes.empty());
    for (double time : buildTimes)
    {
        EXPECT_GT(time, 0.0);
    }
}

TEST(OctreePerformance, ThreadScaling)
{
    std::vector<int> threadCounts = {1, 2, 4, 8, 16};
    if (std::thread::hardware_concurrency() > 16)
    {
        threadCounts.push_back(static_cast<int>(std::thread::hardware_concurrency()));
    }

    const size_t numPoints = 1000000;
    auto coords = generateRandomCoordinates(numPoints);

    bvh2::Box<float> box;
    for (size_t i = 0; i < numPoints; ++i)
    {
        bvh2::Vec3<float> point(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
        box.expand(point);
    }

    std::vector<float> x, y, z;
    splitCoordinates(coords, x, y, z);

    std::vector<KeyType> codes(numPoints);
    tf::Executor executor16(16);
    bvh2::computeSfcKeys(x.data(), y.data(), z.data(), codes.data(), numPoints, box, executor16);
    std::sort(codes.begin(), codes.end());

    std::cout << "\n=== Thread Scaling Test ===\n";
    std::cout << "| # Threads | Build Time (ms) | Speedup |\n";
    std::cout << "|-----------|-----------------|---------|\n";

    double baseTime = 0.0;

    for (int threadCount : threadCounts)
    {
        tf::Executor executor(threadCount);

        cstone::Octree<KeyType> octree(16);
        auto start = std::chrono::high_resolution_clock::now();
        octree.build(codes.data(), codes.data() + codes.size(), executor);
        auto end = std::chrono::high_resolution_clock::now();

        auto buildTime = std::chrono::duration<double, std::milli>(end - start).count();

        double speedup = 1.0;
        if (threadCount == threadCounts.front())
        {
            baseTime = buildTime;
        }
        else
        {
            speedup = baseTime / buildTime;
        }

        std::cout << "| " << std::setw(9) << threadCount << " | "
                  << std::setw(15) << std::fixed << std::setprecision(2) << buildTime << " | "
                  << std::setw(7) << std::fixed << std::setprecision(2) << speedup << " |\n";
    }

    EXPECT_GT(baseTime, 0.0);
}
