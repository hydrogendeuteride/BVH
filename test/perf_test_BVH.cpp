#include <gtest/gtest.h>
#include "bvh/BVH.h"

#include <chrono>
#include <random>
#include <thread>
#include <vector>

using namespace bvh2;

namespace
{

std::vector<Primitive> generateRandomPrimitives(int count, float minCoord = 0.0f, float maxCoord = 100.0f)
{
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(minCoord, maxCoord);

    std::vector<Primitive> primitives(count);
    for (int i = 0; i < count; ++i)
    {
        float center[3] = {dis(gen), dis(gen), dis(gen)};
        float size = dis(gen) * 0.1f + 0.5f;

        primitives[i].bounds.min[0] = center[0] - size;
        primitives[i].bounds.min[1] = center[1] - size;
        primitives[i].bounds.min[2] = center[2] - size;

        primitives[i].bounds.max[0] = center[0] + size;
        primitives[i].bounds.max[1] = center[1] + size;
        primitives[i].bounds.max[2] = center[2] + size;
    }

    return primitives;
}

} // namespace

TEST(BVHPerformance, BuildTime)
{
    for (int count : {10000, 1000000})
    {
        auto primitives = generateRandomPrimitives(count);
        tf::Executor executor{std::thread::hardware_concurrency()};

        auto start = std::chrono::high_resolution_clock::now();
        auto nodes = buildLBVH(executor, primitives);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> elapsed = end - start;
        EXPECT_EQ(nodes.size(), static_cast<size_t>(2 * count - 1));

        std::cout << "Built BVH with " << count << " primitives in " << elapsed.count() << " ms" << std::endl;
    }
}
