#include <gtest/gtest.h>
#include <taskflow/taskflow.hpp>
#include <iostream>
#include <iomanip>

#include "octree/Csarray.h"
#include "octree/Octree.h"

using KeyType = std::uint64_t;
using TreeNodeIndex = int;

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

void traverseOctree(
        const unsigned long *prefixes,
        const cstone::OctreeView<const uint64_t> &view,
        cstone::TreeNodeIndex nodeIdx = 0,
        int depth = 0
)
{
    std::string indent(depth * 2, ' ');

    uint64_t packed = prefixes[nodeIdx];
    uint64_t key = decodePlaceholderBit(packed);
    unsigned lvl = decodePrefixLength(packed) / 3;

    std::cout << indent
              << "[L" << lvl << "] idx=" << nodeIdx
              << "  key=" << key
              << "  prefixLen=" << (lvl * 3)
              << '\n';

    TreeNodeIndex childStart = view.childOffsets[nodeIdx];
    if (childStart == 0) return;

    for (int i = 0; i < 8; ++i)
    {
        TreeNodeIndex childIdx = childStart + i;
        if (childIdx >= view.numNodes) break;
        if (view.parents[(childIdx - 1) / 8] != nodeIdx) continue;

        traverseOctree(prefixes, view, childIdx, depth + 1);
    }
}

TEST(Octree, DebugPrint)
{
    constexpr unsigned bucketSize = 16;
    std::vector<KeyType> codes = makeRandomCodes(100);

    std::size_t nParticles = codes.size();

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

    const auto &prefixes = view.prefixes;
    std::cout << "\n=== Octree Structure ===\n";
    traverseOctree(prefixes, view);

    //cornerstone array generation test?
    EXPECT_EQ(tree.size(), counts.size() + 1);
    EXPECT_EQ(std::accumulate(counts.begin(), counts.end(), size_t(0)), nParticles);
    for (size_t i = 0; i + 1 < tree.size(); ++i)
    {
        EXPECT_LT(tree[i], tree[i + 1]) << "tree not strictly increasing at " << i;
    }

    for (auto c: counts)
    {
        EXPECT_LE(c, bucketSize) << "bucketSize exceeded";
    }

    for (TreeNodeIndex i = 0; i + 1 < view.numNodes; ++i)
    {
        EXPECT_LE(view.prefixes[i], view.prefixes[i + 1])
                            << "prefixes not sorted at " << i;
    }

    //parent-child relation pointer test
    for (TreeNodeIndex pid = 0; pid < view.numInternalNodes; ++pid)
    {
        TreeNodeIndex childStart = view.childOffsets[pid];
        if (childStart == 0) continue;
        for (int j = 0; j < 8; ++j)
        {
            TreeNodeIndex cid = childStart + j;
            if (cid >= view.numNodes) break;
            EXPECT_EQ(view.parents[(cid - 1) / 8], pid)
                                << "parent mismatch: child " << cid;
        }
    }

    //leaf node internal node matching test
    for (TreeNodeIndex lid = 0; lid < view.numLeafNodes; ++lid)
    {
        TreeNodeIndex sortedIdx = view.leafToInternal[lid];
        EXPECT_EQ(view.internalToLeaf[sortedIdx] + view.numInternalNodes,
                  lid)
                            << "leaf↔internal mapping broken at leaf " << lid;
    }
}
