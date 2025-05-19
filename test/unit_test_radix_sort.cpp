#include <gtest/gtest.h>
#include "util/ParallelRadixSort.h"
#include <algorithm>
#include <random>
#include <chrono>

TEST(ChunkedRadixSortTest, BasicSorting)
{
    std::vector<MortonPrimitive<uint32_t>> primitives = {
            {0, 6},
            {1, 1},
            {2, 5},
            {3, 2}
    };

    tf::Executor executor{ std::thread::hardware_concurrency() };
    ChunkedRadixSort(executor, primitives);

    EXPECT_EQ(primitives[0].mortonCode, 1);
    EXPECT_EQ(primitives[1].mortonCode, 2);
    EXPECT_EQ(primitives[2].mortonCode, 5);
    EXPECT_EQ(primitives[3].mortonCode, 6);
}

TEST(ChunkedRadixSortTest, AlreadySorted)
{
    std::vector<MortonPrimitive<uint32_t>> primitives = {
            {0, 0},
            {1, 1},
            {2, 2},
            {3, 3}
    };

    tf::Executor executor{ std::thread::hardware_concurrency() };
    ChunkedRadixSort(executor, primitives);

    EXPECT_EQ(primitives[0].mortonCode, 0);
    EXPECT_EQ(primitives[1].mortonCode, 1);
    EXPECT_EQ(primitives[2].mortonCode, 2);
    EXPECT_EQ(primitives[3].mortonCode, 3);
}

TEST(ChunkedRadixSortTest, EmptyVector)
{
    std::vector<MortonPrimitive<uint32_t>> primitives;

    tf::Executor executor{ std::thread::hardware_concurrency() };
    ChunkedRadixSort(executor, primitives);

    EXPECT_TRUE(primitives.empty());
}

TEST(ChunkedRadixSortTest, SingleElement)
{
    std::vector<MortonPrimitive<uint32_t>> primitives = {{0, 7}};

    tf::Executor executor{ std::thread::hardware_concurrency() };
    ChunkedRadixSort(executor, primitives);

    EXPECT_EQ(primitives[0].mortonCode, 7);
}

TEST(ChunkedRadixSortTest, DuplicatedMortonCodes)
{
    std::vector<MortonPrimitive<uint32_t>> primitives = {
            {0, 2},
            {1, 1},
            {2, 2},
            {3, 3}
    };

    tf::Executor executor{ std::thread::hardware_concurrency() };
    ChunkedRadixSort(executor, primitives);

    EXPECT_EQ(primitives[0].mortonCode, 1);
    EXPECT_EQ(primitives[1].mortonCode, 2);
    EXPECT_EQ(primitives[2].mortonCode, 2);
    EXPECT_EQ(primitives[3].mortonCode, 3);
}

std::vector<MortonPrimitive<uint64_t>> generateRandomMortonPrimitives(size_t count)
{
    std::vector<MortonPrimitive<uint64_t>> primitives;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint64_t> dist(0, 0xFFFFFFFFFFFFFFFFULL);
    for (size_t i = 0; i < count; ++i)
    {
        primitives.push_back({static_cast<uint32_t>(i), dist(rng)});
    }
    return primitives;
}

TEST(ChunkedRadixSortTest, RandomDataSorting)
{
    auto primitives = generateRandomMortonPrimitives(1000000);

    auto expected = primitives;

    auto start = std::chrono::high_resolution_clock::now();
    std::sort(expected.begin(), expected.end(),
              [](const MortonPrimitive<uint64_t> &a, const MortonPrimitive<uint64_t> &b) {
                  return a.mortonCode < b.mortonCode;
              });

    auto end = std::chrono::high_resolution_clock::now();
    auto stdSortDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    tf::Executor executor{ std::thread::hardware_concurrency() };

    start = std::chrono::high_resolution_clock::now();
    ChunkedRadixSort(executor, primitives);
    end = std::chrono::high_resolution_clock::now();
    auto radixDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Radix Sort Duration: " << radixDuration << " ms" << std::endl;
    std::cout << "std::sort Duration: " << stdSortDuration << " ms" << std::endl;

    ASSERT_EQ(primitives.size(), expected.size());
    for (size_t i = 0; i < primitives.size(); ++i)
    {
        EXPECT_EQ(primitives[i].mortonCode, expected[i].mortonCode)
                            << "Mismatch at index " << i;
    }
}
