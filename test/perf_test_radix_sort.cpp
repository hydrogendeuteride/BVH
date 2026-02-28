#include <gtest/gtest.h>
#include "util/ParallelRadixSort.h"

#include <algorithm>
#include <chrono>
#include <limits>
#include <random>
#include <thread>
#include <vector>

using namespace bvh2;

namespace
{

std::vector<MortonPrimitive<uint64_t>> generateRandomMortonPrimitives(size_t count)
{
    std::vector<MortonPrimitive<uint64_t>> primitives;
    primitives.reserve(count);

    std::mt19937_64 rng(42);
    std::uniform_int_distribution<uint64_t> dist(0, std::numeric_limits<uint64_t>::max());

    for (size_t i = 0; i < count; ++i)
    {
        primitives.push_back({static_cast<uint32_t>(i), dist(rng)});
    }

    return primitives;
}

} // namespace

TEST(RadixSortPerformance, RandomDataSorting)
{
    auto primitives = generateRandomMortonPrimitives(1000000);
    auto expected = primitives;

    auto start = std::chrono::high_resolution_clock::now();
    std::sort(expected.begin(), expected.end(),
              [](const MortonPrimitive<uint64_t> &a, const MortonPrimitive<uint64_t> &b) {
                  if (a.mortonCode != b.mortonCode)
                  {
                      return a.mortonCode < b.mortonCode;
                  }

                  return a.primitiveIndex < b.primitiveIndex;
              });
    auto end = std::chrono::high_resolution_clock::now();
    auto stdSortDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    tf::Executor executor{std::thread::hardware_concurrency()};
    start = std::chrono::high_resolution_clock::now();
    ChunkedRadixSort(executor, primitives);
    end = std::chrono::high_resolution_clock::now();
    auto radixDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Radix Sort Duration: " << radixDuration << " ms" << std::endl;
    std::cout << "std::sort Duration: " << stdSortDuration << " ms" << std::endl;

    ASSERT_EQ(primitives.size(), expected.size());
    for (size_t i = 0; i < primitives.size(); ++i)
    {
        EXPECT_EQ(primitives[i], expected[i]) << "Mismatch at index " << i;
    }
}
