#include <gtest/gtest.h>
#include "util/ParallelRadixSort.h"
#include <algorithm>
#include <thread>
#include <vector>

using namespace bvh2;

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
