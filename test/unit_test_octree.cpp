#include <gtest/gtest.h>
#include <taskflow/taskflow.hpp>
#include <iostream>
#include <iomanip>

#include "octree/Csarray.h"
#include "octree/Octree.h"

using KeyType = std::uint64_t;

std::vector<KeyType> makeRandomCodes(std::size_t n, uint64_t seed = 42)
{
    std::mt19937_64 rng(seed);
    std::vector<KeyType> codes(n);

    for (auto &c: codes)
        c = rng() & ((KeyType(1) << 60) - 1);

    std::sort(codes.begin(), codes.end());
    codes.erase(std::unique(codes.begin(), codes.end()), codes.end());
    return codes;
}

TEST(Octree, DebugPrint)
{
    constexpr unsigned bucketSize = 16;
    std::vector<KeyType> codes = makeRandomCodes(100);

    tf::Executor executor;
    cstone::Octree<KeyType> oct(bucketSize);
    oct.build(codes.data(), codes.data() + codes.size(), executor);

    const auto &tree = oct.cornerstone();
    const auto &counts = oct.counts();
    const auto view = oct.view();

    std::cout << "\n== Cornerstone Tree ==\n";
    for (std::size_t i = 0; i < tree.size() - 1; ++i)
    {
        std::cout << "[" << std::setw(2) << i << "] "
                  << "Key: " << tree[i]
                  << " - " << tree[i + 1]
                  << " (count: " << counts[i] << ")\n";
    }

    std::cout << "\n== Prefixes (SFC nodes) ==\n";
    for (cstone::TreeNodeIndex i = 0; i < view.numNodes; ++i)
    {
        KeyType key = view.prefixes[i];
        KeyType decoded = decodePlaceholderBit(key);
        unsigned level = decodePrefixLength(key) / 3;

        std::cout << "[" << std::setw(2) << i << "] "
                  << "Encoded: " << key
                  << " | Decoded: " << decoded
                  << " | Level: " << level << "\n";
    }

    std::cout << "\n== Parent/Child Links ==\n";
    for (cstone::TreeNodeIndex i = 0; i < view.numInternalNodes; ++i)
    {
        std::cout << "[Internal " << i << "] child offset = "
                  << view.childOffsets[i] << "\n";
    }

    for (cstone::TreeNodeIndex i = 0; i < view.numLeafNodes; ++i)
    {
        std::cout << "[Leaf " << i << "] parent = "
                  << view.parents[i / 8] << "\n";
    }
}
