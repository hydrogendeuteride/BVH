#include <gtest/gtest.h>
#include <taskflow/taskflow.hpp>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <string>
#include <vector>

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
    uint64_t key = bvh2::decodePlaceholderBit(packed);
    unsigned lvl = bvh2::decodePrefixLength(packed) / 3;

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
        KeyType decoded = bvh2::decodePlaceholderBit(key);
        unsigned level = bvh2::decodePrefixLength(key) / 3;

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

    std::cout << "\n=== Level info ===\n";
    for (int i = 0; i < 4; ++i)
    {
        int numNodes = oct.view().levelRange[i + 1] - oct.view().levelRange[i];
        if (numNodes == 0)
        { break; }
        std::cout << "number of nodes at level " << i << ": " << numNodes << std::endl;
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

using Vec3f = bvh2::Vec3<float>;

static void checkParentContainsChildren(const std::vector<Vec3f> &centers,
                                        const std::vector<Vec3f> &sizes,
                                        const TreeNodeIndex *childOffsets,
                                        const TreeNodeIndex *parents,
                                        TreeNodeIndex numNodes)
{
    for (TreeNodeIndex parent = 0; parent < numNodes; ++parent)
    {
        TreeNodeIndex firstChild = childOffsets[parent];
        if (firstChild == 0) continue;

        Vec3f pMin = centers[parent] - sizes[parent];
        Vec3f pMax = centers[parent] + sizes[parent];

        for (int i = 0; i < 8; ++i)
        {
            TreeNodeIndex child = firstChild + i;
            if (child >= numNodes) break;
            if (parents[(child - 1) / 8] != parent) continue;

            Vec3f cMin = centers[child] - sizes[child];
            Vec3f cMax = centers[child] + sizes[child];

            EXPECT_LE(pMin.x, cMin.x);
            EXPECT_LE(pMin.y, cMin.y);
            EXPECT_LE(pMin.z, cMin.z);

            EXPECT_GE(pMax.x, cMax.x);
            EXPECT_GE(pMax.y, cMax.y);
            EXPECT_GE(pMax.z, cMax.z);
        }
    }
}

void traverseOctree2(
        const KeyType *prefixes,
        const cstone::OctreeView<const uint64_t> &view,
        const std::vector<Vec3f> &centers,
        const std::vector<Vec3f> &sizes,
        cstone::TreeNodeIndex nodeIdx = 0,
        int depth = 0)
{
    std::string indent(depth * 2, ' ');

    uint64_t packed = prefixes[nodeIdx];
    uint64_t key = bvh2::decodePlaceholderBit(packed);
    unsigned lvl = bvh2::decodePrefixLength(packed) / 3;

    const auto &c = centers[nodeIdx];
    const auto &s = sizes[nodeIdx];

    std::cout << indent
              << "[L" << lvl << "] idx=" << nodeIdx
              << " key=" << key
              << " box=("
              << c.x - s.x << "," << c.y - s.y << "," << c.z - s.z << ") – ("
              << c.x + s.x << "," << c.y + s.y << "," << c.z + s.z << ")\n";

    TreeNodeIndex childStart = view.childOffsets[nodeIdx];
    if (childStart == 0) return;

    for (int i = 0; i < 8; ++i)
    {
        TreeNodeIndex childIdx = childStart + i;
        if (childIdx >= view.numNodes) break;
        if (view.parents[(childIdx - 1) / 8] != nodeIdx) continue;

        traverseOctree2(prefixes, view, centers, sizes, childIdx, depth + 1);
    }
}


TEST(Octree, ParentContainsChildren)
{
    const int N = 100;
    std::vector<float> x(N), y(N), z(N);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    for (int i = 0; i < N; ++i)
    {
        x[i] = uni(rng);
        y[i] = uni(rng);
        z[i] = uni(rng);
    }

    bvh2::Box<float> globalBox(Vec3f{0, 0, 0}, Vec3f{1, 1, 1});

    std::vector<KeyType> keys(N);
    tf::Executor executor;
    bvh2::computeSfcKeys(x.data(), y.data(), z.data(), keys.data(), N, globalBox, executor);

    const unsigned bucketSize = 16;
    cstone::Octree<KeyType> tree(bucketSize);
    tree.build(keys.data(), keys.data() + keys.size(), executor);

    auto view = tree.view();

    std::vector<Vec3f> centers(view.numNodes);
    std::vector<Vec3f> sizes(view.numNodes);
    cstone::nodeFpCenters<KeyType>(view.prefixes, view.numNodes,
                                   centers.data(), sizes.data(),
                                   globalBox, executor);

    checkParentContainsChildren(centers, sizes,
                                view.childOffsets,
                                view.parents,
                                view.numNodes);

    std::cout << "\n=== Tree Hierarchy Test ===" << std::endl;
    traverseOctree2(view.prefixes, view, centers, sizes);
}

TEST(Octree, TraversalVisitsAllNodes)
{
    constexpr unsigned bucketSize = 16;
    std::vector<KeyType> codes = makeRandomCodes(256);

    tf::Executor executor;
    cstone::Octree<KeyType> oct(bucketSize);
    oct.build(codes.data(), codes.data() + codes.size(), executor);

    auto view = oct.view();

    std::vector<TreeNodeIndex> visited;
    cstone::traverseOctree(view, [&](TreeNodeIndex idx, KeyType key, unsigned level) {
        (void)key;
        (void)level;
        visited.push_back(idx);
        return true;
    });

    ASSERT_EQ(static_cast<TreeNodeIndex>(visited.size()), view.numNodes);

    std::sort(visited.begin(), visited.end());
    visited.erase(std::unique(visited.begin(), visited.end()), visited.end());
    EXPECT_EQ(static_cast<TreeNodeIndex>(visited.size()), view.numNodes);
}
