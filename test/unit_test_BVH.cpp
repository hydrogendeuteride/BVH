#include <gtest/gtest.h>
#include "bvh/BVH.h"
#include "bvh/Reflection.h"
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <thread>

using namespace bvh2;

class BVHTest : public ::testing::Test
{
protected:
    std::vector<Primitive> generateRandomPrimitives(int count, float minCoord = -0.0f, float maxCoord = 100.0f)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
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

    bool contains(const BoundingBox &outer, const BoundingBox &inner)
    {
        for (int i = 0; i < 3; i++)
        {
            if (outer.min[i] > inner.min[i] || outer.max[i] < inner.max[i])
            {
                return false;
            }
        }
        return true;
    }

    bool validateBVHStructure(const BVHData<float> &nodes)
    {
        if (nodes.empty()) return true;

        std::vector<bool> visited(nodes.size(), false);
        std::vector<uint32_t> pending{0};
        while (!pending.empty())
        {
            const uint32_t i = pending.back(); pending.pop_back();
            if (i >= nodes.size() || visited[i]) return false;
            visited[i] = true;
            const auto& node = nodes[i];
            if (!node.isLeaf())
            {
                if (node.left_idx >= nodes.size() || node.right_idx() >= nodes.size()) return false;
                if (!contains(node.bounds, nodes[node.left_idx].bounds) ||
                    !contains(node.bounds, nodes[node.right_idx()].bounds)) return false;
                pending.push_back(node.left_idx);
                pending.push_back(node.right_idx());
            }
        }
        return std::all_of(visited.begin(), visited.end(), [](bool v) { return v; });
    }

    bool validateBoundingBoxes(const BVHData<float> &nodes, const std::vector<Primitive> &primitives)
    {
        if (nodes.empty()) return primitives.empty();

        for (size_t index = 0; index < nodes.size(); ++index)
        {
            const auto node = nodes[index];
            if (node.isLeaf())
            {
                if (node.object_idx >= primitives.size())
                {
                    return false;
                }

                const BoundingBox &nodeBounds = node.bounds;
                const BoundingBox &primBounds = primitives[node.object_idx].bounds;

                for (int i = 0; i < 3; i++)
                {
                    if (std::abs(nodeBounds.min[i] - primBounds.min[i]) > 1e-5f ||
                        std::abs(nodeBounds.max[i] - primBounds.max[i]) > 1e-5f)
                    {
                        return false;
                    }
                }
            }
        }

        for (size_t i = 0; i < nodes.size(); i++)
        {
            const BVHNode &node = nodes[i];

            if (!node.isLeaf())
            {
                const BVHNode &leftChild = nodes[node.left_idx];
                const BVHNode &rightChild = nodes[node.right_idx()];

                BoundingBox unionBox;
                for (int j = 0; j < 3; j++)
                {
                    unionBox.min[j] = std::min(leftChild.bounds.min[j], rightChild.bounds.min[j]);
                    unionBox.max[j] = std::max(leftChild.bounds.max[j], rightChild.bounds.max[j]);
                }

                for (int j = 0; j < 3; j++)
                {
                    if (std::abs(node.bounds.min[j] - unionBox.min[j]) > 1e-5f ||
                        std::abs(node.bounds.max[j] - unionBox.max[j]) > 1e-5f)
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    void SetUp() override
    {

    }

    void TearDown() override
    {

    }
};

TEST_F(BVHTest, EmptyInput)
{
    std::vector<Primitive> emptyPrimitives;
    tf::Executor executor{ std::thread::hardware_concurrency() };
    auto nodes = buildLBVH<uint32_t>(executor, emptyPrimitives);
    EXPECT_TRUE(nodes.empty());
}

TEST_F(BVHTest, SinglePrimitive)
{
    Primitive p;
    p.bounds.min[0] = p.bounds.min[1] = p.bounds.min[2] = -1.0f;
    p.bounds.max[0] = p.bounds.max[1] = p.bounds.max[2] = 1.0f;

    std::vector<Primitive> primitives = {p};
    tf::Executor executor{ std::thread::hardware_concurrency() };
    auto nodes = buildLBVH(executor, primitives);

    EXPECT_EQ(nodes.size(), 1);
    EXPECT_TRUE(nodes[0].isLeaf());
    EXPECT_EQ(nodes[0].object_idx, 0u);

    for (int i = 0; i < 3; i++)
    {
        EXPECT_NEAR(nodes[0].bounds.min[i], -1.0f, 1e-5f);
        EXPECT_NEAR(nodes[0].bounds.max[i], 1.0f, 1e-5f);
    }
}

TEST_F(BVHTest, SimpleCase)
{
    Primitive p1, p2;

    p1.bounds.min[0] = p1.bounds.min[1] = p1.bounds.min[2] = 1.0f;
    p1.bounds.max[0] = p1.bounds.max[1] = p1.bounds.max[2] = 0.0f;

    p2.bounds.min[0] = p2.bounds.min[1] = p2.bounds.min[2] = 0.0f;
    p2.bounds.max[0] = p2.bounds.max[1] = p2.bounds.max[2] = 2.0f;

    std::vector<Primitive> primitives = {p1, p2};
    tf::Executor executor{ std::thread::hardware_concurrency() };
    auto nodes = buildLBVH(executor, primitives);

    EXPECT_EQ(nodes.size(), 3);

    EXPECT_FALSE(nodes[0].isLeaf());
    EXPECT_TRUE(nodes[1].isLeaf());
    EXPECT_TRUE(nodes[2].isLeaf());

    EXPECT_TRUE(validateBoundingBoxes(nodes, primitives));
    EXPECT_TRUE(validateBVHStructure(nodes));
}

TEST_F(BVHTest, SortedPrimitives)
{
    std::vector<Primitive> primitives;
    const int count = 10;

    for (int i = 0; i < count; i++)
    {
        Primitive p;
        float pos = i * 2.0f;
        p.bounds.min[0] = pos - 0.5f;
        p.bounds.min[1] = p.bounds.min[2] = -0.5f;
        p.bounds.max[0] = pos + 0.5f;
        p.bounds.max[1] = p.bounds.max[2] = 0.5f;
        primitives.push_back(p);
    }

    tf::Executor executor{ std::thread::hardware_concurrency() };
    auto nodes = buildLBVH(executor, primitives);

    EXPECT_EQ(nodes.size(), 2 * count - 1);

    EXPECT_TRUE(validateBoundingBoxes(nodes, primitives));
    EXPECT_TRUE(validateBVHStructure(nodes));
}

TEST_F(BVHTest, FindSplit)
{
    std::vector<MortonPrimitive<uint32_t>> mortonPrimitives(10);

    for (int i = 0; i < 10; i++)
    {
        mortonPrimitives[i].primitiveIndex = i;
        mortonPrimitives[i].mortonCode = i * 1000;
    }

    uint32_t split = findSplit(mortonPrimitives, 10, 0, 9);

    EXPECT_GT(split, 0);
    EXPECT_LT(split, 9);
}

TEST_F(BVHTest, RandomPrimitives)
{
    for (int count: {10, 100})
    {
        auto primitives = generateRandomPrimitives(count);
        tf::Executor executor{ std::thread::hardware_concurrency() };

        auto nodes = buildLBVH(executor, primitives);

        EXPECT_EQ(nodes.size(), 2 * count - 1);

        EXPECT_TRUE(validateBoundingBoxes(nodes, primitives));
        EXPECT_TRUE(validateBVHStructure(nodes));
    }
}

TEST_F(BVHTest, AllPrimitivesAtSamePosition)
{
    const int count = 10;
    std::vector<Primitive> primitives(count);

    for (int i = 0; i < count; i++)
    {
        primitives[i].bounds.min[0] = primitives[i].bounds.min[1] = primitives[i].bounds.min[2] = 0.0f;
        primitives[i].bounds.max[0] = primitives[i].bounds.max[1] = primitives[i].bounds.max[2] = 1.0f;
    }

    tf::Executor executor{ std::thread::hardware_concurrency() };
    auto nodes = buildLBVH(executor, primitives);

    EXPECT_EQ(nodes.size(), 2 * count - 1);

    EXPECT_TRUE(validateBVHStructure(nodes));
    EXPECT_TRUE(validateBoundingBoxes(nodes, primitives));
}

TEST_F(BVHTest, RayTraversalMissesBVH)
{
    std::vector<Primitive> primitives(1);
    primitives[0].bounds.min[0] = primitives[0].bounds.min[1] = primitives[0].bounds.min[2] = 0.0f;
    primitives[0].bounds.max[0] = primitives[0].bounds.max[1] = primitives[0].bounds.max[2] = 1.0f;

    // Define a triangle on one face of the box (not actually hit in this test).
    primitives[0].v0 = Vec3<float>(0.0f, 0.0f, 0.0f);
    primitives[0].v1 = Vec3<float>(0.0f, 1.0f, 0.0f);
    primitives[0].v2 = Vec3<float>(0.0f, 0.0f, 1.0f);
    primitives[0].updateBounds();

    tf::Executor executor{ std::thread::hardware_concurrency() };
    auto nodes = buildLBVH(executor, primitives);

    Ray ray(Vec3<float>(2.0f, 2.0f, 2.0f), Vec3<float>(1.0f, 0.0f, 0.0f));

    uint32_t hitIndex = 0;
    float hitT = 0.0f;
    bool hit = traverseBVHClosestHit(nodes, primitives, ray, hitIndex, hitT);

    EXPECT_FALSE(hit);
}

TEST_F(BVHTest, RayTraversalFindsClosestPrimitive)
{
    std::vector<Primitive> primitives(2);

    // First primitive at [0,1]^3
    primitives[0].bounds.min[0] = primitives[0].bounds.min[1] = primitives[0].bounds.min[2] = 0.0f;
    primitives[0].bounds.max[0] = primitives[0].bounds.max[1] = primitives[0].bounds.max[2] = 1.0f;

    // Triangle on the front face x = 0 so the ray hits at t = 1.
    primitives[0].v0 = Vec3<float>(0.0f, 0.0f, 0.0f);
    primitives[0].v1 = Vec3<float>(0.0f, 1.0f, 0.0f);
    primitives[0].v2 = Vec3<float>(0.0f, 0.0f, 1.0f);
    primitives[0].updateBounds();

    // Second primitive at [2,3]^3
    primitives[1].bounds.min[0] = primitives[1].bounds.min[1] = primitives[1].bounds.min[2] = 2.0f;
    primitives[1].bounds.max[0] = primitives[1].bounds.max[1] = primitives[1].bounds.max[2] = 3.0f;

    // Triangle on the front face x = 2 so the ray hits later (t = 3).
    primitives[1].v0 = Vec3<float>(2.0f, 0.0f, 0.0f);
    primitives[1].v1 = Vec3<float>(2.0f, 1.0f, 0.0f);
    primitives[1].v2 = Vec3<float>(2.0f, 0.0f, 1.0f);
    primitives[1].updateBounds();

    tf::Executor executor{ std::thread::hardware_concurrency() };
    auto nodes = buildLBVH(executor, primitives);

    Ray ray(Vec3<float>(-1.0f, 0.5f, 0.5f), Vec3<float>(1.0f, 0.0f, 0.0f));

    uint32_t hitIndex = 0;
    float hitT = 0.0f;
    bool hit = traverseBVHClosestHit(nodes, primitives, ray, hitIndex, hitT);

    EXPECT_TRUE(hit);
    EXPECT_EQ(hitIndex, 0u);
    EXPECT_GT(hitT, 0.0f);
}

TEST_F(BVHTest, RayAABBParallelAxisBoundaryIsStable)
{
    BoundingBox box(Vec3<float>(0.0f, 0.0f, 0.0f), Vec3<float>(1.0f, 1.0f, 1.0f));
    Ray ray(Vec3<float>(0.0f, 0.5f, 0.5f), Vec3<float>(0.0f, 1.0f, 0.0f), 0.0f, 10.0f);

    float tNear = 0.0f;
    float tFar = 0.0f;
    bool hit = intersectRayAABB(ray, box, ray.tmin, ray.tmax, tNear, tFar);

    EXPECT_TRUE(hit);
    EXPECT_TRUE(std::isfinite(tNear));
    EXPECT_TRUE(std::isfinite(tFar));
    EXPECT_LE(tNear, tFar);
}

TEST_F(BVHTest, RayAABBParallelAxisOutsideSlabMisses)
{
    BoundingBox box(Vec3<float>(0.0f, 0.0f, 0.0f), Vec3<float>(1.0f, 1.0f, 1.0f));
    Ray ray(Vec3<float>(-0.1f, 0.5f, 0.5f), Vec3<float>(0.0f, 1.0f, 0.0f), 0.0f, 10.0f);

    float tNear = 0.0f;
    float tFar = 0.0f;
    bool hit = intersectRayAABB(ray, box, ray.tmin, ray.tmax, tNear, tFar);

    EXPECT_FALSE(hit);
}

TEST_F(BVHTest, BuildLBVHWithDifferentSortMethods)
{
    auto primitives = generateRandomPrimitives(128);

    tf::Executor executor{ std::thread::hardware_concurrency() };

    auto nodesStd = buildLBVH<uint64_t>(executor, primitives, MortonSortMethod::StdSort);
    auto nodesRadix = buildLBVH<uint64_t>(executor, primitives, MortonSortMethod::RadixSort);

    EXPECT_EQ(nodesStd.size(), nodesRadix.size());
    EXPECT_TRUE(validateBVHStructure(nodesStd));
    EXPECT_TRUE(validateBVHStructure(nodesRadix));
}

TEST_F(BVHTest, BuildersUseEveryNodeAndPrimitiveOnce)
{
    tf::Executor executor(4);
    for (int count : {0, 1, 2, 3, 9, 128, 4096}) {
        const auto primitives = generateRandomPrimitives(count);
        for (const auto& nodes : {buildSAHBVH(executor, primitives),
                                 buildLBVH<uint32_t>(executor, primitives),
                                 buildLBVH<uint64_t>(executor, primitives),
                                 buildLBVH<uint64_t>(executor, primitives, MortonSortMethod::RadixSort)}) {
            ASSERT_EQ(nodes.size(), count ? 2u * count - 1 : 0u);
            ASSERT_TRUE(validateBVHStructure(nodes));
            ASSERT_TRUE(validateBoundingBoxes(nodes, primitives));
            std::vector<unsigned> seen(count, 0);
            traverseBVH(nodes, [&](uint32_t, const BVHNode& node) {
                if (node.isLeaf()) ++seen.at(node.object_idx);
            });
            for (auto visits : seen) EXPECT_EQ(visits, 1u);
        }
    }
}

TEST_F(BVHTest, ClosestHitMatchesBruteForce)
{
    auto check = [](auto scalar) {
        using S = decltype(scalar);
        using V = Vec3<S>;
        std::vector<PrimitiveT<S>> triangles;
        for (int i = 0; i < 33; ++i)
            triangles.emplace_back(V(i, -1, -1), V(i, 1, -1), V(i, 0, 1));
        tf::Executor executor(2);
        for (const auto& nodes : {buildSAHBVH(executor, triangles), buildLBVH(executor, triangles)}) {
        for (S dx : {S(-1), S(0), std::numeric_limits<S>::denorm_min(), S(1)})
            for (S dy : {S(0), S(.01), S(2)}) for (S start : {S(-1), S(16), S(34)}) {
                RayT<S> ray(V(start, 0, 0), V(dx, dy, S(.001)), S(0), S(1));
                bool expected = false;
                S nearest = ray.tmax;
                for (const auto& triangle : triangles) {
                    S t;
                    if (intersectRayPrimitive(ray, triangle, ray.tmin, nearest, t)) {
                        expected = true; nearest = t;
                    }
                }
                uint32_t id = UINT32_MAX;
                S t = -1;
                ASSERT_EQ(traverseBVHClosestHit(nodes, triangles, ray, id, t), expected);
                if (expected) { EXPECT_EQ(t, nearest); EXPECT_LT(id, triangles.size()); }
            }
        }
    };
    check(float{});
    check(double{});
}

TEST_F(BVHTest, SoAViewsAndTraversal)
{
    tf::Executor executor(2);
    auto check = [&](auto scalar) {
        using S = decltype(scalar);
        using V = Vec3<S>;
        const std::vector<PrimitiveT<S>> triangles{
            {V(0, -1, -1), V(0, 1, -1), V(0, 0, 1)},
            {V(1, -1, -1), V(1, 1, -1), V(1, 0, 1)}};
        for (auto tree : {buildSAHBVH(executor, triangles), buildLBVH(executor, triangles)}) {
            auto view = tree.data();
            const auto constView = std::as_const(tree).data();
            static_assert(std::is_same_v<decltype(constView.bounds), const Box<S>*>);
            static_assert(std::is_same_v<decltype(constView.childOffsets), const uint32_t*>);
            ASSERT_EQ(view.numNodes, 3u);
            EXPECT_EQ(view.bounds, tree.bounds.data());
            EXPECT_EQ(view.childOffsets, tree.childOffsets.data());
            EXPECT_EQ(view.primitiveIndices, tree.primitiveIndices.data());
            const RayT<S> ray(V(S(.5), 0, 0), V(1, 0, 0));
            uint32_t id = UINT32_MAX;
            S t = -1;
            ASSERT_TRUE(traverseBVHClosestHit(view, triangles, ray, id, t));
            EXPECT_EQ(id, 1u);
            EXPECT_EQ(t, S(.5));
            ASSERT_TRUE(traverseBVHClosestHit(constView, triangles, ray, id, t));
            EXPECT_EQ(id, 1u);
            ASSERT_TRUE(traverseBVHClosestNodeHit(tree, ray, 0, id, t));
            EXPECT_EQ(id, 0u);
            EXPECT_EQ(t, S(0));
            ASSERT_TRUE(traverseBVHClosestNodeHit(view, ray, 1, id, t));
            EXPECT_EQ(view.primitiveIndices[id], 1u);
            EXPECT_EQ(t, S(.5));
            unsigned visited = 0;
            traverseBVH(constView, [&](uint32_t, const auto&) { ++visited; return false; });
            EXPECT_EQ(visited, 1u);
            unsigned bounces = 0;
            const auto count = traceBVHReflections(view, triangles, ray, 4,
                [&](const ReflectionHitT<S>& hit) { EXPECT_EQ(hit.primitiveIndex, 1u - bounces++ % 2); });
            EXPECT_EQ(count, 4u);
            EXPECT_EQ(bounces, 4u);
            EXPECT_EQ(traceBVHReflections(tree, triangles, ray, 4,
                [](const ReflectionHitT<S>&) { return false; }), 1u);
        }
        BVHData<S> empty;
        uint32_t id = 99;
        S t = 42;
        const RayT<S> ray(V(0, 0, 0), V(1, 0, 0));
        EXPECT_FALSE(traverseBVHClosestHit(empty.data(), triangles, ray, id, t));
        EXPECT_FALSE(traverseBVHClosestNodeHit(empty.data(), ray, 0, id, t));
        EXPECT_EQ(id, 99u);
        EXPECT_EQ(t, S(42));
    };
    check(float{});
    check(double{});
}

TEST_F(BVHTest, SiblingAABBMatchesSeparateTests)
{
    auto check = [](auto scalar) {
        using S = decltype(scalar);
        using V = Vec3<S>;
        std::mt19937 random(42);
        std::uniform_real_distribution<S> uniform(-4, 4);
        auto compare = [](const RayT<S>& ray, const Box<S>* boxes) {
            const detail::PreparedRayT<S> prepared(ray);
            S near[2] = {};
            const unsigned mask = detail::intersectPreparedRayAABBPair(
                prepared, boxes, ray.tmin, ray.tmax, near[0], near[1]);
            for (int child = 0; child < 2; ++child) {
                S expectedNear, expectedFar;
                const bool hit = detail::intersectPreparedRayAABB(
                    prepared, boxes[child], ray.tmin, ray.tmax, expectedNear, expectedFar);
                ASSERT_EQ(bool(mask & (1u << child)), hit);
                if (hit) { EXPECT_EQ(near[child], expectedNear); }
            }
        };
        for (int trial = 0; trial < 512; ++trial) {
            Box<S> boxes[2];
            for (auto& box : boxes) for (int axis = 0; axis < 3; ++axis) {
                box.min[axis] = uniform(random);
                box.max[axis] = box.min[axis] + std::abs(uniform(random));
            }
            RayT<S> ray(V(uniform(random), uniform(random), uniform(random)),
                        V(uniform(random), uniform(random), uniform(random)), S(0), S(10));
            compare(ray, boxes);
            for (S direction : {S(0), -S(0), std::numeric_limits<S>::denorm_min()}) {
                ray.direction[trial % 3] = direction;
                ray.origin[trial % 3] = boxes[0].min[trial % 3];
                compare(ray, boxes);
            }
        }
        Box<S> boxes[2];
        for (auto& box : boxes) { box.min = V(-1, -1, -1); box.max = V(1, 1, 1); }
        RayT<S> ray(V(0, 0, 0), V(1, 1, 1), S(0), S(1));
        for (S invalid : {std::numeric_limits<S>::infinity(), std::numeric_limits<S>::quiet_NaN()}) {
            boxes[0].min.x = invalid;
            compare(ray, boxes); // An invalid sibling must not hide the valid sibling.
            boxes[0].min.x = -1;
            ray.origin.x = invalid;
            compare(ray, boxes);
            ray.origin.x = 0;
            ray.tmax = invalid;
            compare(ray, boxes);
            ray.tmax = 1;
        }
        ray.tmin = 2;
        compare(ray, boxes);
    };
    check(float{});
    check(double{});
}

TEST_F(BVHTest, LBVHBoundsMatchAcrossWorkersAndSorts)
{
    tf::Executor serial(1), parallel(4);
    auto primitives = generateRandomPrimitives(8192);
    for (size_t i = 1; i < primitives.size(); i += 3) primitives[i] = primitives[0];
    const auto expected = buildLBVH<uint64_t>(serial, primitives, MortonSortMethod::StdSort);
    for (auto method : {MortonSortMethod::StdSort, MortonSortMethod::RadixSort}) {
        const auto actual = buildLBVH<uint64_t>(parallel, primitives, method);
        ASSERT_EQ(actual.size(), expected.size());
        EXPECT_EQ(actual.childOffsets, expected.childOffsets);
        EXPECT_EQ(actual.primitiveIndices, expected.primitiveIndices);
        ASSERT_TRUE(validateBVHStructure(actual));
        for (size_t i = 0; i < actual.size(); ++i) for (int axis = 0; axis < 3; ++axis) {
            EXPECT_EQ(actual.bounds[i].min[axis], expected.bounds[i].min[axis]);
            EXPECT_EQ(actual.bounds[i].max[axis], expected.bounds[i].max[axis]);
        }
    }
}

TEST_F(BVHTest, SAHBinBoundariesMatchBruteForce)
{
    tf::Executor executor(4);
    auto check = [&](auto scalar) {
        using S = decltype(scalar);
        using V = Vec3<S>;
        std::vector<PrimitiveT<S>> triangles;
        for(int i=0;i<=32;++i) {
            const S x=S(i)/S(16)-S(1);
            for(S position : {std::nextafter(x,-std::numeric_limits<S>::infinity()), x,
                              std::nextafter(x,std::numeric_limits<S>::infinity())})
                triangles.emplace_back(V(position,-1,-1),V(position,1,-1),V(position,0,1));
        }
        const auto tree=buildSAHBVH(executor,triangles);
        for(S start : {S(-2),S(-.75),S(0),S(.5),S(2)}) for(S direction : {S(-1),S(1)}) {
            const RayT<S> ray(V(start,0,0),V(direction,0,0),S(0),S(4));
            bool expected=false;
            S nearest=ray.tmax;
            for(const auto& triangle:triangles) {
                S t;
                if(intersectRayPrimitive(ray,triangle,ray.tmin,nearest,t)) { expected=true; nearest=t; }
            }
            uint32_t id=UINT32_MAX;
            S t=-1;
            ASSERT_EQ(traverseBVHClosestHit(tree,triangles,ray,id,t),expected);
            if(expected) { EXPECT_EQ(t,nearest); EXPECT_LT(id,triangles.size()); }
        }
    };
    check(float{});
    check(double{});
}

TEST_F(BVHTest, SAHLargeAndOverlappingRanges)
{
    tf::Executor serial(1), parallel(4);
    auto check = [&](auto scalar) {
        using S = decltype(scalar);
        using V = Vec3<S>;
        for (S spread : {S(1), S(.001)}) {
            std::mt19937 random(42);
            std::uniform_real_distribution<S> uniform(-spread, spread);
            std::vector<PrimitiveT<S>> triangles;
            for (unsigned i = 0; i < 20000; ++i) {
                const V a(uniform(random), uniform(random), uniform(random));
                triangles.emplace_back(a, a + V(S(.1), S(.03), 0), a + V(0, S(.1), S(.03)));
            }
            const auto tree = buildSAHBVH(serial, triangles);
            const auto threaded = buildSAHBVH(parallel, triangles);
            ASSERT_EQ(tree.size(), 2 * triangles.size() - 1);
            EXPECT_EQ(tree.childOffsets, threaded.childOffsets);
            EXPECT_EQ(tree.primitiveIndices, threaded.primitiveIndices);
            std::vector<unsigned> visits(triangles.size());
            size_t visited = 0;
            traverseBVH(tree, [&](uint32_t index, const auto& node) {
                ++visited;
                for (int axis = 0; axis < 3; ++axis) {
                    EXPECT_EQ(node.bounds.min[axis], threaded.bounds[index].min[axis]);
                    EXPECT_EQ(node.bounds.max[axis], threaded.bounds[index].max[axis]);
                }
                if (node.isLeaf()) { ++visits.at(node.object_idx); }
                else for (uint32_t child : {node.left_idx, node.right_idx()})
                    for (int axis = 0; axis < 3; ++axis) {
                        EXPECT_GE(tree.bounds[child].min[axis], node.bounds.min[axis]);
                        EXPECT_LE(tree.bounds[child].max[axis], node.bounds.max[axis]);
                    }
            });
            EXPECT_EQ(visited, tree.size());
            EXPECT_TRUE(std::all_of(visits.begin(), visits.end(), [](unsigned n) { return n == 1; }));
            for (unsigned i = 0; i < 32; ++i) {
                const auto& triangle = triangles[i * 613];
                V target = triangle.v0 + triangle.v1 + triangle.v2;
                for (int axis = 0; axis < 3; ++axis) target[axis] /= S(3);
                const RayT<S> ray(target + V(0, 0, 2), V(0, 0, -1));
                S nearest = ray.tmax;
                bool expected = false;
                for (const auto& primitive : triangles) {
                    S t;
                    if (intersectRayPrimitive(ray, primitive, ray.tmin, nearest, t)) { expected = true; nearest = t; }
                }
                uint32_t id = UINT32_MAX;
                S t = -1;
                ASSERT_EQ(traverseBVHClosestHit(tree, triangles, ray, id, t), expected);
                ASSERT_TRUE(expected);
                EXPECT_EQ(t, nearest);
            }
        }
    };
    check(float{});
    check(double{});
}
