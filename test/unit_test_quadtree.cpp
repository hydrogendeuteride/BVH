#include <gtest/gtest.h>
#include <taskflow/taskflow.hpp>
#include <algorithm>
#include <numeric>
#include <random>
#include <vector>
#include <iostream>
#include <iomanip>

#include "quadtree/Quadtree.h"
#include "quadtree/Hilbert2D.h"
#include "util/Box2D.h"
#include "quadtree/Bitops2D.h"

using KeyType = std::uint64_t;
using TreeNodeIndex = qtree2d::TreeNodeIndex;

static std::vector<float> generateRandom2D(size_t n, unsigned seed = 42)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> uni(0.0f, 1.0f);
    std::vector<float> xy(n * 2);
    for (size_t i = 0; i < n; ++i)
    {
        xy[2 * i + 0] = uni(rng);
        xy[2 * i + 1] = uni(rng);
    }
    return xy;
}

static void splitXY(const std::vector<float> &xy, std::vector<float> &x, std::vector<float> &y)
{
    size_t n = xy.size() / 2;
    x.resize(n); y.resize(n);
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = xy[2 * i + 0];
        y[i] = xy[2 * i + 1];
    }
}

static void checkParentContainsChildren(const std::vector<Vec2<float>> &centers,
                                        const std::vector<Vec2<float>> &sizes,
                                        const TreeNodeIndex *childOffsets,
                                        const TreeNodeIndex *parents,
                                        TreeNodeIndex numNodes)
{
    const float eps = 1e-6f;
    for (TreeNodeIndex parent = 0; parent < numNodes; ++parent)
    {
        TreeNodeIndex firstChild = childOffsets[parent];
        if (firstChild == 0) continue;

        Vec2<float> pMin = centers[parent] - sizes[parent];
        Vec2<float> pMax = centers[parent] + sizes[parent];

        for (int i = 0; i < 4; ++i)
        {
            TreeNodeIndex child = firstChild + i;
            if (child >= numNodes) break;
            if (parents[(child - 1) / 4] != parent) continue;

            Vec2<float> cMin = centers[child] - sizes[child];
            Vec2<float> cMax = centers[child] + sizes[child];

            EXPECT_LE(pMin.x, cMin.x + eps);
            EXPECT_LE(pMin.y, cMin.y + eps);
            EXPECT_GE(pMax.x + eps, cMax.x);
            EXPECT_GE(pMax.y + eps, cMax.y);
        }
    }
}

static void traverseQuadtree(const KeyType *prefixes,
                             const qtree2d::QuadtreeView<const KeyType> &view,
                             qtree2d::TreeNodeIndex nodeIdx = 0,
                             int depth = 0,
                             int maxDepth = -1)
{
    if (maxDepth >= 0 && depth > maxDepth) return;
    std::string indent(depth * 2, ' ');

    KeyType packed = prefixes[nodeIdx];
    KeyType key = decodePlaceholderBit2D<KeyType>(packed);
    unsigned lvl = decodePrefixLength2D<KeyType>(packed) / 2;

    std::cout << indent
              << "[L" << lvl << "] idx=" << nodeIdx
              << " key=" << key
              << '\n';

    TreeNodeIndex childStart = view.childOffsets[nodeIdx];
    if (childStart == 0) return;

    for (int i = 0; i < 4; ++i)
    {
        TreeNodeIndex childIdx = childStart + i;
        if (childIdx >= view.numNodes) break;
        if (view.parents[(childIdx - 1) / 4] != nodeIdx) continue;

        traverseQuadtree(prefixes, view, childIdx, depth + 1, maxDepth);
    }
}

static void printParentChildTable(const qtree2d::QuadtreeView<const KeyType> &view)
{
    std::cout << "\n== Parent/Child (quadtree) ==\n";
    for (TreeNodeIndex pid = 0; pid < view.numInternalNodes; ++pid)
    {
        TreeNodeIndex childStart = view.childOffsets[pid];
        if (childStart == 0) continue;
        unsigned plevel = decodePrefixLength2D<KeyType>(view.prefixes[pid]) / 2;
        std::cout << "P[" << pid << "] L" << plevel << " -> ";
        for (int j = 0; j < 4; ++j)
        {
            TreeNodeIndex cid = childStart + j;
            if (cid >= view.numNodes) break;
            if (view.parents[(cid - 1) / 4] != pid) continue;
            unsigned clevel = decodePrefixLength2D<KeyType>(view.prefixes[cid]) / 2;
            std::cout << "C(" << cid << ":L" << clevel << ") ";
        }
        std::cout << '\n';
    }
}

TEST(Quadtree2D, BuildBasic)
{
    const size_t N = 1000;
    auto xy = generateRandom2D(N, 12345);
    std::vector<float> xs, ys; splitXY(xy, xs, ys);

    Box2D<float> box;
    for (size_t i = 0; i < N; ++i) box.expand({xs[i], ys[i]});
    // small padding
    float rx = box.width() * 0.01f, ry = box.height() * 0.01f;
    box.min.x -= rx; box.max.x += rx; box.min.y -= ry; box.max.y += ry;

    std::vector<KeyType> keys(N);
    tf::Executor ex(1);
    computeSfcKeys2D<float, KeyType>(xs.data(), ys.data(), keys.data(), N, box, ex);
    std::sort(keys.begin(), keys.end());

    const unsigned bucketSize = 16;
    qtree2d::Quadtree<KeyType> qt(bucketSize);
    qt.build(keys.data(), keys.data() + keys.size(), ex);

    const auto &tree = qt.cornerstone();
    const auto &counts = qt.counts();
    const auto view = qt.view();

    // cornerstone invariants
    ASSERT_EQ(tree.size(), counts.size() + 1);
    ASSERT_EQ(std::accumulate(counts.begin(), counts.end(), size_t(0)), N);
    for (size_t i = 0; i + 1 < tree.size(); ++i) EXPECT_LT(tree[i], tree[i + 1]);
    for (auto c: counts) EXPECT_LE(c, bucketSize);

    // prefixes sorted
    for (TreeNodeIndex i = 0; i + 1 < view.numNodes; ++i)
        EXPECT_LE(view.prefixes[i], view.prefixes[i + 1]);

    // parent-child linkage
    for (TreeNodeIndex pid = 0; pid < view.numInternalNodes; ++pid)
    {
        TreeNodeIndex childStart = view.childOffsets[pid];
        if (childStart == 0) continue;
        for (int j = 0; j < 4; ++j)
        {
            TreeNodeIndex cid = childStart + j;
            if (cid >= view.numNodes) break;
            EXPECT_EQ(view.parents[(cid - 1) / 4], pid);
        }
    }

    // leaf <-> internal mapping
    for (TreeNodeIndex lid = 0; lid < view.numLeafNodes; ++lid)
    {
        TreeNodeIndex sortedIdx = view.leafToInternal[lid];
        EXPECT_EQ(view.internalToLeaf[sortedIdx] + view.numInternalNodes, lid);
    }
}

TEST(Quadtree2D, ParentContainsChildren)
{
    const int N = 200;
    auto xy = generateRandom2D(N, 2025);
    std::vector<float> xs, ys; splitXY(xy, xs, ys);

    Box2D<float> box;
    for (int i = 0; i < N; ++i) box.expand({xs[i], ys[i]});

    std::vector<KeyType> keys(N);
    tf::Executor ex;
    computeSfcKeys2D<float, KeyType>(xs.data(), ys.data(), keys.data(), N, box, ex);
    std::sort(keys.begin(), keys.end());

    const unsigned bucket = 8;
    qtree2d::Quadtree<KeyType> qt(bucket);
    qt.build(keys.data(), keys.data() + keys.size(), ex);

    auto view = qt.view();
    std::vector<Vec2<float>> centers(view.numNodes);
    std::vector<Vec2<float>> sizes(view.numNodes);
    qtree2d::nodeFpCenters2D<KeyType>(view.prefixes, view.numNodes,
                                      centers.data(), sizes.data(), box, ex);

    checkParentContainsChildren(centers, sizes,
                                view.childOffsets, view.parents, view.numNodes);

    // Debug output: parent-child relations and a shallow traversal
    printParentChildTable(view);
    std::cout << "\n=== Quadtree Structure (depth<=2) ===\n";
    traverseQuadtree(view.prefixes, view, 0, 0, 2);
}

TEST(Quadtree2D, EdgeSmall)
{
    tf::Executor ex;
    // N = 1
    {
        Box2D<float> box({0, 0}, {1, 1});
        std::vector<KeyType> keys = {iHilbert2D<KeyType>(0, 0)};
        qtree2d::Quadtree<KeyType> qt(16);
        qt.build(keys.data(), keys.data() + keys.size(), ex);
        auto v = qt.view();
        EXPECT_EQ(v.numLeafNodes, 1);
        EXPECT_EQ(v.numInternalNodes, 0);
        EXPECT_EQ(v.childOffsets[0], 0);
    }

    // N = 2
    {
        std::vector<KeyType> keys = {iHilbert2D<KeyType>(0, 0), iHilbert2D<KeyType>(1, 0)};
        std::sort(keys.begin(), keys.end());
        qtree2d::Quadtree<KeyType> qt(1);
        qt.build(keys.data(), keys.data() + keys.size(), ex);
        auto v = qt.view();
        EXPECT_GE(v.numLeafNodes, 2);
        EXPECT_GE(v.numNodes, 2);
    }
}

TEST(Quadtree2D, TraversalVisitsAllNodes)
{
    const size_t N = 512;
    auto xy = generateRandom2D(N, 777);
    std::vector<float> xs, ys;
    splitXY(xy, xs, ys);

    Box2D<float> box;
    for (size_t i = 0; i < N; ++i) box.expand({xs[i], ys[i]});

    std::vector<KeyType> keys(N);
    tf::Executor ex(1);
    computeSfcKeys2D<float, KeyType>(xs.data(), ys.data(), keys.data(), N, box, ex);
    std::sort(keys.begin(), keys.end());

    const unsigned bucketSize = 16;
    qtree2d::Quadtree<KeyType> qt(bucketSize);
    qt.build(keys.data(), keys.data() + keys.size(), ex);

    auto view = qt.view();

    std::vector<TreeNodeIndex> visited;
    qtree2d::traverseQuadtree(view, [&](TreeNodeIndex idx, KeyType key, unsigned level) {
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
