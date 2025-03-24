#ifndef BVH2_BVH_H
#define BVH2_BVH_H

#include "BoundingBox.h"
#include "MortonCode.h"
#include "ParallelRadixSort.h"
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

//    std::sort(mortonPrimitives.begin(), mortonPrimitives.end(),
//              [](const MortonPrimitive &a, const MortonPrimitive &b) {
//                  return a.mortonCode < b.mortonCode;
//              });

    ChunkedRadixSort(mortonPrimitives);

    return mortonPrimitives;
}

std::vector<BVHNode>
buildBVH(const std::vector<Primitive> &primitives, const std::vector<MortonPrimitive> &mortonPrimitives)
{
    uint32_t numPrimitives = static_cast<uint32_t>(primitives.size());
    uint32_t numInternalNodes = numPrimitives - 1;
    uint32_t totalNodes = numPrimitives + numInternalNodes;

    std::vector<BVHNode> nodes(totalNodes);

#pragma omp parallel for
    for (size_t i = 0; i < totalNodes; ++i)
    {
        nodes[i].object_idx = (i >= numInternalNodes) ? i - numInternalNodes : 0xFFFFFFFF;
        nodes[i].parent_idx = 0;
        nodes[i].isLeaf = (i >= numInternalNodes);
    }

#pragma omp parallel for
    for (uint32_t idx = 0; idx < numInternalNodes; ++idx)
    {
        BVHNode &node = nodes[idx];

        const PrimitiveRange range = determineRange(idx, numPrimitives, mortonPrimitives);
        const int gamma = findSplit(mortonPrimitives, numPrimitives, range.first, range.last);

        nodes[idx].left_idx = gamma;
        nodes[idx].right_idx = gamma + 1;

        if (std::min(range.first, range.last) == gamma)
        {
            nodes[idx].left_idx += numInternalNodes;
        }
        if (std::max(range.first, range.last) == gamma + 1)
        {
            nodes[idx].right_idx += numInternalNodes;
        }

        nodes[nodes[idx].left_idx].parent_idx = idx;
        nodes[nodes[idx].right_idx].parent_idx = idx;
    }

#pragma omp parallel for
    for (uint32_t idx = 0; idx < numPrimitives; ++idx)
    {
        BVHNode &node = nodes[idx + numInternalNodes];
        uint32_t primIdx = mortonPrimitives[idx].primitiveIndex;
        node.object_idx = primIdx;
        node.bounds = primitives[primIdx].bounds;
    }

    //-------------------------------------------------------------------------------------------------------------
    std::vector<std::atomic<int>> flags(numInternalNodes);
    for (uint32_t i = 0; i < numInternalNodes; ++i)
    {
        flags[i].store(0);
    }

#pragma omp parallel for
    for (uint32_t idx = numInternalNodes; idx < totalNodes; ++idx)
    {
        uint32_t parent = nodes[idx].parent_idx;

        while (parent != 0 || flags[0].load() != 0)
        {
            int expected = 0;
            bool wasFirst = flags[parent].compare_exchange_strong(expected, 1);

            if (wasFirst)
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

                uint32_t nextParent = parentNode.parent_idx;
                parent = nextParent;
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

    return nodes;
}

std::vector<BVHNode> buildLBVH(const std::vector<Primitive> &primitives)
{
    if (primitives.empty()) return {};

    std::vector<MortonPrimitive> mortonPrimitives = generateMortonCodes(primitives);

    return buildBVH(primitives, mortonPrimitives);
}

#endif //BVH2_BVH_H
