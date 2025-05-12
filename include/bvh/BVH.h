#ifndef BVH2_BVH_H
#define BVH2_BVH_H

#include "util/BoundingBox.h"
#include "util/MortonCode.h"
#include "util/ParallelRadixSort.h"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <vector>
#include <algorithm>
#include <atomic>

struct Primitive
{
    BoundingBox bounds;
};

struct BVHNode
{
    BoundingBox bounds;
    uint32_t object_idx;      //leaf nodes: index of the primitive; internal node: 0xFFFFFFFF
    uint32_t left_idx;
    uint32_t right_idx;
    uint32_t parent_idx;
    bool isLeaf;
};

inline uint32_t
findSplit(const std::vector<MortonPrimitive> &mortonPrimitives, const uint32_t numPrimitives, uint32_t first,
          uint32_t last)
{
    if (first == last)
    {
        return first;
    }

    uint64_t firstCode = mortonPrimitives[first].mortonCode;
    uint64_t lastCode = mortonPrimitives[last].mortonCode;

    if (firstCode == lastCode)
    {
        return (first + last) >> 1;
    }

    uint32_t commonPrefix = __builtin_clzll(firstCode ^ lastCode);

    uint32_t split = first;
    uint32_t step = last - first;

    do
    {
        step = (step + 1) >> 1;
        uint32_t newSplit = split + step;

        if (newSplit < last)
        {
            uint64_t splitCode = mortonPrimitives[newSplit].mortonCode;
            uint32_t splitPrefix = __builtin_clzll(firstCode ^ splitCode);

            if (splitPrefix > commonPrefix)
            {
                split = newSplit;
            }
        }
    } while (step > 1);

    return split;
}

struct PrimitiveRange
{
    uint32_t first;
    uint32_t last;
};

inline int commonUpperBits(uint64_t a, uint64_t b)
{
    return __builtin_clzll(a ^ b);
}

inline PrimitiveRange
determineRange(uint32_t idx, uint32_t numPrimitives, const std::vector<MortonPrimitive> &mortonPrimitives)
{
    if (idx == 0)
    {
        return {0, numPrimitives - 1};
    }

    uint64_t mortonCode = mortonPrimitives[idx].mortonCode;

    const int L_delta = (idx > 0) ? commonUpperBits(mortonCode, mortonPrimitives[idx - 1].mortonCode) : -1;

    const int R_delta = (idx < numPrimitives - 1) ? commonUpperBits(mortonCode, mortonPrimitives[idx + 1].mortonCode)
                                                  : -1;

    const int d = (R_delta > L_delta) ? 1 : -1;

    const int delta_min = std::min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = idx + d * l_max;

    if (0 <= i_tmp && i_tmp < static_cast<int>(numPrimitives))
    {
        delta = commonUpperBits(mortonCode, mortonPrimitives[i_tmp].mortonCode);
    }

    while (delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = idx + d * l_max;
        delta = -1;

        if (0 <= i_tmp && i_tmp < static_cast<int>(numPrimitives))
        {
            delta = commonUpperBits(mortonCode, mortonPrimitives[i_tmp].mortonCode);
        }
    }

    int l = 0;
    int t = l_max >> 1;
    while (t > 0)
    {
        i_tmp = idx + (l + t) * d;
        delta = -1;
        if (0 <= i_tmp && i_tmp < static_cast<int>(numPrimitives))
        {
            delta = commonUpperBits(mortonCode, mortonPrimitives[i_tmp].mortonCode);
        }
        if (delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }

    unsigned int jdx = idx + l * d;
    if (d < 0)
    {
        std::swap(idx, jdx);
    }

    return {idx, jdx};
}

std::vector<MortonPrimitive> generateMortonCodes(const std::vector<Primitive> &primitives)
{
    BoundingBox sceneBounds;
    for (const auto &prim: primitives)
    {
        float centeroid[3];
        prim.bounds.centroid(centeroid);
        sceneBounds.expand(centeroid);
    }

    float sceneMin[3], sceneExtent[3];
    for (int i = 0; i < 3; ++i)
    {
        sceneMin[i] = sceneBounds.min[i];
        sceneExtent[i] = sceneBounds.max[i] - sceneBounds.min[i];

        if (sceneExtent[i] < 1e-6f) sceneExtent[i] = 1e-6f;
    }

    std::vector<MortonPrimitive> mortonPrimitives(primitives.size());
    for (int i = 0; i < primitives.size(); ++i)
    {
        float centroid[3];
        primitives[i].bounds.centroid(centroid);

        mortonPrimitives[i].primitiveIndex = static_cast<uint32_t>(i);
        mortonPrimitives[i].mortonCode = computeMortonCode(centroid, sceneMin, sceneExtent);
    }

    std::sort(mortonPrimitives.begin(), mortonPrimitives.end(),
              [](const MortonPrimitive &a, const MortonPrimitive &b) {
                  return a.mortonCode < b.mortonCode;
              });

//    ChunkedRadixSort(mortonPrimitives);

    return mortonPrimitives;
}

std::vector<BVHNode>
buildBVH(tf::Executor &executor, const std::vector<Primitive> &primitives, const std::vector<MortonPrimitive> &mortonPrimitives)
{
    uint32_t numPrimitives = static_cast<uint32_t>(primitives.size());

    if (numPrimitives == 1)
    {
        std::vector<BVHNode> nodes(1);
        nodes[0].isLeaf = true;
        nodes[0].object_idx = mortonPrimitives[0].primitiveIndex;
        nodes[0].bounds = primitives[mortonPrimitives[0].primitiveIndex].bounds;
        nodes[0].parent_idx = 0;
        return nodes;
    }

    uint32_t numInternalNodes = numPrimitives - 1;
    uint32_t totalNodes = numPrimitives + numInternalNodes;

    std::vector<BVHNode> nodes(totalNodes);

    tf::Taskflow tf;

    auto taskInit = tf.for_each_index(0u, totalNodes, 1u, [&](uint32_t i) {
        nodes[i].object_idx = (i >= numInternalNodes) ? i - numInternalNodes : 0xFFFFFFFF;
        nodes[i].parent_idx = 0;
        nodes[i].isLeaf = (i >= numInternalNodes);
    });

    auto taskInternal = tf.for_each_index(0u, numInternalNodes, 1u, [&](uint32_t idx) {
        BVHNode &node = nodes[idx];

        const PrimitiveRange range = determineRange(idx, numPrimitives, mortonPrimitives);
        const uint32_t gamma = findSplit(mortonPrimitives, numPrimitives, range.first, range.last);

        node.left_idx = gamma;
        node.right_idx = gamma + 1;

        if (std::min(range.first, range.last) == gamma) node.left_idx += numInternalNodes;
        if (std::max(range.first, range.last) == gamma + 1) node.right_idx += numInternalNodes;

        nodes[node.left_idx].parent_idx = idx;
        nodes[node.right_idx].parent_idx = idx;
    });

    auto taskLeaf = tf.for_each_index(0u, numPrimitives, 1u, [&](uint32_t idx) {
        BVHNode &node = nodes[idx + numInternalNodes];
        uint32_t pIdx = mortonPrimitives[idx].primitiveIndex;
        node.object_idx = pIdx;
        node.bounds = primitives[pIdx].bounds;
    });

    //-------------------------------------------------------------------------------------------------------------
    auto taskBounds = tf.emplace([&]() {
        std::vector<std::atomic<int>> flags(numInternalNodes);
        for (uint32_t i = 0; i < numInternalNodes; ++i) flags[i].store(0);

        for (uint32_t idx = numInternalNodes; idx < totalNodes; ++idx)
        {
            uint32_t parent = nodes[idx].parent_idx;

            while (parent != 0 || flags[0].load() != 0)
            {
                int expected = 0;
                bool first = flags[parent].compare_exchange_strong(expected, 1);

                if (first)
                {
                    break;
                }
                else
                {
                    BVHNode &parentNode = nodes[parent];
                    BVHNode &leftChild = nodes[parentNode.left_idx];
                    BVHNode &rightChild = nodes[parentNode.right_idx];

                    for (int j = 0; j < 3; ++j)
                    {
                        parentNode.bounds.min[j] = std::min(leftChild.bounds.min[j], rightChild.bounds.min[j]);
                        parentNode.bounds.max[j] = std::max(leftChild.bounds.max[j], rightChild.bounds.max[j]);
                    }

                    parent = parentNode.parent_idx;
                }
            }
        }

        if (numInternalNodes > 0 && flags[0].load() == 0)
        {
            BVHNode &rootNode = nodes[0];
            BVHNode &leftChild = nodes[rootNode.left_idx];
            BVHNode &rightChild = nodes[rootNode.right_idx];

            for (int j = 0; j < 3; ++j)
            {
                rootNode.bounds.min[j] = std::min(leftChild.bounds.min[j], rightChild.bounds.min[j]);
                rootNode.bounds.max[j] = std::max(leftChild.bounds.max[j], rightChild.bounds.max[j]);
            }
        }
    });

    taskInit.precede(taskInternal, taskLeaf);
    taskInternal.precede(taskBounds);
    taskLeaf.precede(taskBounds);

    executor.run(tf).wait();

    return nodes;
}

std::vector<BVHNode> buildLBVH(tf::Executor &executor, const std::vector<Primitive> &primitives)
{
    if (primitives.empty()) return {};

    std::vector<MortonPrimitive> mortonPrimitives = generateMortonCodes(primitives);

    return buildBVH(executor, primitives, mortonPrimitives);
}

#endif //BVH2_BVH_H
