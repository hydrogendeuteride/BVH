#include <gtest/gtest.h>
#include "ParallelRadixSort.h"

TEST(ChunkedRadixSortTest, BasicSorting) {
    std::vector<MortonPrimitive> primitives = {
            {0, 6},
            {1, 1},
            {2, 5},
            {3, 2}
    };

    ChunkedRadixSort(primitives);

    EXPECT_EQ(primitives[0].mortonCode, 1);
    EXPECT_EQ(primitives[1].mortonCode, 2);
    EXPECT_EQ(primitives[2].mortonCode, 5);
    EXPECT_EQ(primitives[3].mortonCode, 6);
}

TEST(ChunkedRadixSortTest, AlreadySorted) {
    std::vector<MortonPrimitive> primitives = {
            {0, 0},
            {1, 1},
            {2, 2},
            {3, 3}
    };

    ChunkedRadixSort(primitives);

    EXPECT_EQ(primitives[0].mortonCode, 0);
    EXPECT_EQ(primitives[1].mortonCode, 1);
    EXPECT_EQ(primitives[2].mortonCode, 2);
    EXPECT_EQ(primitives[3].mortonCode, 3);
}

TEST(ChunkedRadixSortTest, EmptyVector) {
    std::vector<MortonPrimitive> primitives;

    ChunkedRadixSort(primitives);

    EXPECT_TRUE(primitives.empty());
}

TEST(ChunkedRadixSortTest, SingleElement) {
    std::vector<MortonPrimitive> primitives = { {0, 7} };

    ChunkedRadixSort(primitives);

    EXPECT_EQ(primitives[0].mortonCode, 7);
}

TEST(ChunkedRadixSortTest, DuplicatedMortonCodes) {
    std::vector<MortonPrimitive> primitives = {
            {0, 2},
            {1, 1},
            {2, 2},
            {3, 3}
    };

    ChunkedRadixSort(primitives);

    EXPECT_EQ(primitives[0].mortonCode, 1);
    EXPECT_EQ(primitives[1].mortonCode, 2);
    EXPECT_EQ(primitives[2].mortonCode, 2);
    EXPECT_EQ(primitives[3].mortonCode, 3);
}