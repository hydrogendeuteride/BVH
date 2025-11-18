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

std::vector<float> generateRandomCoordinates(size_t numPoints, float min = -100.0f, float max = 100.0f)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(min, max);

    std::vector<float> coords(numPoints * 3);
    for (size_t i = 0; i < coords.size(); ++i)
    {
        coords[i] = dis(gen);
    }
    return coords;
}

void
splitCoordinates(const std::vector<float> &coords, std::vector<float> &x, std::vector<float> &y, std::vector<float> &z)
{
    size_t numPoints = coords.size() / 3;
    x.resize(numPoints);
    y.resize(numPoints);
    z.resize(numPoints);
    for (size_t i = 0; i < numPoints; ++i)
    {
        x[i] = coords[i * 3];
        y[i] = coords[i * 3 + 1];
        z[i] = coords[i * 3 + 2];
    }
}

// Hilbert 코드로 변환
std::vector<KeyType> generateHilbertCodes(const std::vector<float> &coords, const Box<float> &box)
{
    size_t numPoints = coords.size() / 3;
    std::vector<KeyType> codes(numPoints);

    for (size_t i = 0; i < numPoints; ++i)
    {
        float x = coords[i * 3];
        float y = coords[i * 3 + 1];
        float z = coords[i * 3 + 2];
        codes[i] = hilbert3D<KeyType>(x, y, z, box);
    }

    return codes;
}

TEST(OctreePerformance, BuildTime)
{
    std::vector<size_t> pointCounts = {1000, 10000, 100000, 1000000};

    std::vector<double> buildTimes;
    std::vector<double> perPointTimes;

    tf::Executor executor(1);

    std::cout << "\n=== Octree Build Performance Test ===\n";
    std::cout << "| #Points | Build Time (ms) | Time per Point (ns) |\n";
    std::cout << "|---------|-----------------|---------------------|\n";

    for (size_t numPoints: pointCounts)
    {
        auto coords = generateRandomCoordinates(numPoints);

        Box<float> box;
        for (size_t i = 0; i < numPoints; ++i)
        {
            Vec3<float> point(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
            box.expand(point);
        }

        for (int i = 0; i < 3; ++i)
        {
            float range = box.max[i] - box.min[i];
            box.min[i] -= range * 0.01f;
            box.max[i] += range * 0.01f;
        }

        std::vector<float> x, y, z;
        splitCoordinates(coords, x, y, z);

        auto start = std::chrono::high_resolution_clock::now();
        std::vector<KeyType> codes(numPoints);
        computeSfcKeys(x.data(), y.data(), z.data(), codes.data(), numPoints, box, executor);
        auto codeEnd = std::chrono::high_resolution_clock::now();

        std::sort(codes.begin(), codes.end());
        auto sortEnd = std::chrono::high_resolution_clock::now();

        unsigned bucketSize = 16;
        cstone::Octree<KeyType> octree(bucketSize);
        auto buildStart = std::chrono::high_resolution_clock::now();
        octree.build(codes.data(), codes.data() + codes.size(), executor);
        auto buildEnd = std::chrono::high_resolution_clock::now();

        auto codeTime = std::chrono::duration<double, std::milli>(codeEnd - start).count();
        auto sortTime = std::chrono::duration<double, std::milli>(sortEnd - codeEnd).count();
        auto octreeTime = std::chrono::duration<double, std::milli>(buildEnd - buildStart).count();
        auto totalTime = std::chrono::duration<double, std::milli>(buildEnd - start).count();

        double timePerPoint = (totalTime * 1e6) / numPoints;

        buildTimes.push_back(totalTime);
        perPointTimes.push_back(timePerPoint);

        std::cout << "| " << std::setw(7) << numPoints << " | "
                  << std::setw(15) << std::fixed << std::setprecision(2) << totalTime << " | "
                  << std::setw(19) << std::fixed << std::setprecision(2) << timePerPoint << " |\n";

        std::cout << "  - Key generation: " << std::fixed << std::setprecision(2) << codeTime << " ms\n";
        std::cout << "  - Key sorting:    " << std::fixed << std::setprecision(2) << sortTime << " ms\n";
        std::cout << "  - Octree build:   " << std::fixed << std::setprecision(2) << octreeTime << " ms\n";

        const auto &tree = octree.cornerstone();
        const auto &counts = octree.counts();
        const auto view = octree.view();

        std::cout << "  - Node count: " << view.numNodes << " (internal: " << view.numInternalNodes
                  << ", leaf: " << view.numLeafNodes << ")\n";

        double avgFill = 0.0;
        if (!counts.empty())
        {
            avgFill = static_cast<double>(numPoints) / counts.size();
        }
        std::cout << "  - Avg. bucket fill: " << std::fixed << std::setprecision(2) << avgFill
                  << " / " << bucketSize << " (" << (avgFill / bucketSize * 100) << "%)\n";

        std::cout << "\n";
    }

    if (buildTimes.size() >= 2)
    {
        std::cout << "=== Scalability Analysis ===\n";
        for (size_t i = 1; i < pointCounts.size(); ++i)
        {
            double ratio = pointCounts[i] / static_cast<double>(pointCounts[i - 1]);
            double timeRatio = buildTimes[i] / buildTimes[i - 1];

            std::cout << pointCounts[i - 1] << " → " << pointCounts[i] << " points: "
                      << "Data x" << std::fixed << std::setprecision(1) << ratio << ", "
                      << "Time x" << std::fixed << std::setprecision(2) << timeRatio << "\n";

            double linearRatio = ratio;
            double logLinearRatio = ratio * std::log2(pointCounts[i]) / std::log2(pointCounts[i - 1]);

            std::cout << "  - Linear expected (O(n)):    " << std::fixed << std::setprecision(2) << linearRatio
                      << "x\n";
            std::cout << "  - Log-linear expected (O(n log n)): " << std::fixed << std::setprecision(2)
                      << logLinearRatio << "x\n";
        }
    }

    EXPECT_FALSE(buildTimes.empty());
    for (double time: buildTimes)
    {
        EXPECT_GT(time, 0.0);
    }
}

TEST(OctreePerformance, ThreadScaling)
{
    std::vector<int> threadCounts = {1, 2, 4, 8, 16};
    if (std::thread::hardware_concurrency() > 16)
    {
        threadCounts.push_back(std::thread::hardware_concurrency());
    }

    const size_t numPoints = 1000000;
    auto coords = generateRandomCoordinates(numPoints);

    Box<float> box;
    for (size_t i = 0; i < numPoints; ++i)
    {
        Vec3<float> point(coords[i * 3], coords[i * 3 + 1], coords[i * 3 + 2]);
        box.expand(point);
    }

    std::vector<float> x, y, z;
    splitCoordinates(coords, x, y, z);

    std::vector<KeyType> codes(numPoints);
    tf::Executor executor16(16);
    computeSfcKeys(x.data(), y.data(), z.data(), codes.data(), numPoints, box, executor16);
    std::sort(codes.begin(), codes.end());

    std::cout << "\n=== Thread Scaling Test ===\n";
    std::cout << "| # Threads | Build Time (ms) | Speed Improved |\n";
    std::cout << "|-----------|----------------|----------|\n";

    double baseTime = 0.0;

    for (int threadCount: threadCounts)
    {
        tf::Executor executor(threadCount);

        cstone::Octree<KeyType> octree(16);

        auto start = std::chrono::high_resolution_clock::now();
        octree.build(codes.data(), codes.data() + codes.size(), executor);
        auto end = std::chrono::high_resolution_clock::now();

        auto buildTime = std::chrono::duration<double, std::milli>(end - start).count();

        double speedup = 1.0;
        if (threadCount == threadCounts[0])
        {
            baseTime = buildTime;
        }
        else
        {
            speedup = baseTime / buildTime;
        }

        std::cout << "| " << std::setw(9) << threadCount << " | "
                  << std::setw(14) << std::fixed << std::setprecision(2) << buildTime << " | "
                  << std::setw(10) << std::fixed << std::setprecision(2) << speedup << " |\n";
    }

    std::cout << "\nTheoretical Maximum (Amdahl's Law : 90% Parallelization):\n";
    for (int threadCount: threadCounts)
    {
        if (threadCount > 1)
        {
            double theoreticalSpeedup = 1.0 / (0.1 + 0.9 / threadCount);
            std::cout << threadCount << " Thread: " << std::fixed << std::setprecision(2) << theoreticalSpeedup
                      << " Times Speedup\n";
        }
    }

    EXPECT_GT(baseTime, 0.0);
}

using Vec3f = Vec3<float>;

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
    uint64_t key = decodePlaceholderBit(packed);
    unsigned lvl = decodePrefixLength(packed) / 3;

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

    Box<float> globalBox(Vec3f{0, 0, 0}, Vec3f{1, 1, 1});

    std::vector<KeyType> keys(N);
    tf::Executor executor;
    computeSfcKeys(x.data(), y.data(), z.data(), keys.data(), N, globalBox, executor);

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
