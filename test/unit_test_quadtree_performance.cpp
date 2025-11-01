#include <gtest/gtest.h>
#include <taskflow/taskflow.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "quadtree/Quadtree.h"
#include "quadtree/Hilbert2D.h"
#include "util/Box2D.h"

using KeyType = std::uint64_t;

static std::vector<float> gen2D(size_t n, float minv = -100.f, float maxv = 100.f)
{
    std::mt19937 rng(123456u);
    std::uniform_real_distribution<float> d(minv, maxv);
    std::vector<float> xy(n * 2);
    for (size_t i = 0; i < n; ++i) { xy[2 * i + 0] = d(rng); xy[2 * i + 1] = d(rng); }
    return xy;
}

static void splitXY(const std::vector<float> &xy, std::vector<float> &x, std::vector<float> &y)
{
    size_t n = xy.size() / 2; x.resize(n); y.resize(n);
    for (size_t i = 0; i < n; ++i) { x[i] = xy[2 * i + 0]; y[i] = xy[2 * i + 1]; }
}

TEST(QuadtreePerformance, BuildTime)
{
    std::vector<size_t> pointCounts = {1000, 10000, 100000, 1000000};

    tf::Executor executor(1);
    std::cout << "\n=== Quadtree Build Performance Test (2D) ===\n";
    std::cout << "| #Points | Total (ms) | KeyGen (ms) | Sort (ms) | Build (ms) |\n";
    std::cout << "|---------|------------|-------------|-----------|------------|\n";

    for (size_t n : pointCounts)
    {
        auto xy = gen2D(n);
        std::vector<float> xs, ys; splitXY(xy, xs, ys);

        Box2D<float> box;
        for (size_t i = 0; i < n; ++i) box.expand({xs[i], ys[i]});

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<KeyType> keys(n);
        computeSfcKeys2D<float, KeyType>(xs.data(), ys.data(), keys.data(), n, box, executor);
        auto t1 = std::chrono::high_resolution_clock::now();

        std::sort(keys.begin(), keys.end());
        auto t2 = std::chrono::high_resolution_clock::now();

        qtree2d::Quadtree<KeyType> qt(16);
        qt.build(keys.data(), keys.data() + keys.size(), executor);
        auto t3 = std::chrono::high_resolution_clock::now();

        auto keyMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        auto sortMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        auto buildMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
        auto totalMs = std::chrono::duration<double, std::milli>(t3 - t0).count();

        std::cout << "| " << std::setw(7) << n
                  << " | " << std::setw(10) << std::fixed << std::setprecision(2) << totalMs
                  << " | " << std::setw(11) << keyMs
                  << " | " << std::setw(9) << sortMs
                  << " | " << std::setw(10) << buildMs << " |\n";

        // quick sanity
        const auto &counts = qt.counts();
        ASSERT_EQ(std::accumulate(counts.begin(), counts.end(), size_t(0)), n);
    }
}

TEST(QuadtreePerformance, ThreadScaling)
{
    std::vector<int> threadCounts = {1, 2, 4, 8, 16};
    if (std::thread::hardware_concurrency() > 16)
        threadCounts.push_back(std::thread::hardware_concurrency());

    const size_t n = 1000000;
    auto xy = gen2D(n);
    std::vector<float> xs, ys; splitXY(xy, xs, ys);

    Box2D<float> box;
    for (size_t i = 0; i < n; ++i) box.expand({xs[i], ys[i]});

    std::vector<KeyType> keys(n);
    tf::Executor ex16(16);
    computeSfcKeys2D<float, KeyType>(xs.data(), ys.data(), keys.data(), n, box, ex16);
    std::sort(keys.begin(), keys.end());

    std::cout << "\n=== Quadtree Thread Scaling (2D) ===\n";
    std::cout << "| #Threads | Build (ms) | Speedup vs base |\n";
    std::cout << "|----------|-----------|-----------------|\n";

    double base = 0.0;
    for (int tc : threadCounts)
    {
        tf::Executor ex(tc);
        qtree2d::Quadtree<KeyType> qt(16);
        auto t0 = std::chrono::high_resolution_clock::now();
        qt.build(keys.data(), keys.data() + keys.size(), ex);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        if (tc == 1) base = ms;
        double speedup = (tc == 1 || base <= 0.0) ? 1.0 : (base / ms);
        std::cout << "| " << std::setw(8) << tc
                  << " | " << std::setw(9) << std::fixed << std::setprecision(2) << ms
                  << " | " << std::setw(15) << std::fixed << std::setprecision(2) << speedup << " |\n";
    }

    ASSERT_GT(base, 0.0);
}

