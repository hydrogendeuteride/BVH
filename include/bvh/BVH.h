#ifndef BVH2_BVH_H
#define BVH2_BVH_H

#include "util/BoundingBox.h"
#include "util/MortonCode.h"
#include "util/ParallelRadixSort.h"
#include "util/Bitops.h"
#include "util/ray.h"
#include "util/Triangle.h"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <vector>
#include <algorithm>
#include <atomic>
#include <type_traits>

template<typename Scalar>
using PrimitiveT = TriangleT<Scalar>;

using Primitive = PrimitiveT<float>;
using PrimitiveF = PrimitiveT<float>;
using PrimitiveD = PrimitiveT<double>;

template<typename Scalar>
struct BVHNodeT
{
    Box<Scalar> bounds;
    uint32_t object_idx;      //leaf nodes: index of the primitive; internal node: 0xFFFFFFFF
    uint32_t left_idx;
    uint32_t right_idx;
    uint32_t parent_idx;
    bool isLeaf;
};

using BVHNode = BVHNodeT<float>;
using BVHNodeF = BVHNodeT<float>;
using BVHNodeD = BVHNodeT<double>;

enum class MortonSortMethod
{
    StdSort,
    RadixSort
};

template<typename MortonCodeType>
inline uint32_t
findSplit(const std::vector<MortonPrimitive<MortonCodeType>> &mortonPrimitives, const uint32_t numPrimitives,
          uint32_t first,
          uint32_t last)
{
    if (first == last)
    {
        return first;
    }

    MortonCodeType firstCode = mortonPrimitives[first].mortonCode;
    MortonCodeType lastCode = mortonPrimitives[last].mortonCode;

    if (firstCode == lastCode)
    {
        return (first + last) >> 1;
    }

    uint32_t commonPrefix = countLeadingZeros(firstCode ^ lastCode);

    uint32_t split = first;
    uint32_t step = last - first;

    do
    {
        step = (step + 1) >> 1;
        uint32_t newSplit = split + step;

        if (newSplit < last)
        {
            MortonCodeType splitCode = mortonPrimitives[newSplit].mortonCode;
            uint32_t splitPrefix = countLeadingZeros(firstCode ^ splitCode);

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

template<typename MortonCodeType>
inline int commonUpperBits(MortonCodeType a, MortonCodeType b)
{
    return countLeadingZeros(a ^ b);
}

template<typename MortonCodeType>
inline PrimitiveRange
determineRange(uint32_t idx, uint32_t numPrimitives,
               const std::vector<MortonPrimitive<MortonCodeType>> &mortonPrimitives)
{
    if (idx == 0)
    {
        return {0, numPrimitives - 1};
    }

    MortonCodeType mortonCode = mortonPrimitives[idx].mortonCode;

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

template<typename MortonCodeType>
inline void sortMortonPrimitives(std::vector<MortonPrimitive<MortonCodeType>> &mortonPrimitives,
                                 MortonSortMethod method,
                                 tf::Executor *executor)
{
    if (method == MortonSortMethod::RadixSort && executor != nullptr)
    {
        ChunkedRadixSort(*executor, mortonPrimitives);
    }
    else
    {
        std::sort(mortonPrimitives.begin(), mortonPrimitives.end(),
                  [](const MortonPrimitive<MortonCodeType> &a, const MortonPrimitive<MortonCodeType> &b) {
                      return a.mortonCode < b.mortonCode;
                  });
    }
}

template<typename MortonCodeType = uint64_t, typename Scalar>
std::vector<MortonPrimitive<MortonCodeType>>
generateMortonCodes(const std::vector<PrimitiveT<Scalar>> &primitives,
                    MortonSortMethod sortMethod,
                    tf::Executor *executor)
{
    Box<Scalar> sceneBounds;
    for (const auto &prim: primitives)
    {
        Scalar centroid[3];
        prim.bounds.centroid(centroid);
        sceneBounds.expand(centroid);
    }

    Scalar sceneMin[3], sceneExtent[3];
    for (int i = 0; i < 3; ++i)
    {
        sceneMin[i] = sceneBounds.min[i];
        sceneExtent[i] = sceneBounds.max[i] - sceneBounds.min[i];

        if (sceneExtent[i] < static_cast<Scalar>(1e-6)) sceneExtent[i] = static_cast<Scalar>(1e-6);
    }

    std::vector<MortonPrimitive<MortonCodeType>> mortonPrimitives(primitives.size());
    for (size_t i = 0; i < primitives.size(); ++i)
    {
        Scalar centroid[3];
        primitives[i].bounds.centroid(centroid);

        mortonPrimitives[i].primitiveIndex = static_cast<uint32_t>(i);
        mortonPrimitives[i].mortonCode =
                computeMortonCode<MortonCodeType, Scalar>(centroid, sceneMin, sceneExtent);
    }

    sortMortonPrimitives(mortonPrimitives, sortMethod, executor);

    return mortonPrimitives;
}

template<typename MortonCodeType = uint64_t, typename Scalar>
std::vector<MortonPrimitive<MortonCodeType>> generateMortonCodes(const std::vector<PrimitiveT<Scalar>> &primitives)
{
    return generateMortonCodes<MortonCodeType, Scalar>(primitives, MortonSortMethod::StdSort, nullptr);
}

template<typename MortonCodeType, typename Scalar>
std::vector<BVHNodeT<Scalar>>
buildBVH(tf::Executor &executor, const std::vector<PrimitiveT<Scalar>> &primitives,
         const std::vector<MortonPrimitive<MortonCodeType>> &mortonPrimitives)
{
    uint32_t numPrimitives = static_cast<uint32_t>(primitives.size());

    if (numPrimitives == 1)
    {
        std::vector<BVHNodeT<Scalar>> nodes(1);
        nodes[0].isLeaf = true;
        nodes[0].object_idx = mortonPrimitives[0].primitiveIndex;
        nodes[0].bounds = primitives[mortonPrimitives[0].primitiveIndex].bounds;
        nodes[0].parent_idx = 0;
        return nodes;
    }

    uint32_t numInternalNodes = numPrimitives - 1;
    uint32_t totalNodes = numPrimitives + numInternalNodes;

    std::vector<BVHNodeT<Scalar>> nodes(totalNodes);

    tf::Taskflow tf;

    auto taskInit = tf.for_each_index(0u, totalNodes, 1u, [&](uint32_t i) {
        nodes[i].object_idx = (i >= numInternalNodes) ? i - numInternalNodes : 0xFFFFFFFF;
        nodes[i].parent_idx = 0;
        nodes[i].isLeaf = (i >= numInternalNodes);
    });

    auto taskInternal = tf.for_each_index(0u, numInternalNodes, 1u, [&](uint32_t idx) {
        BVHNodeT<Scalar> &node = nodes[idx];

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
        BVHNodeT<Scalar> &node = nodes[idx + numInternalNodes];
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
                    BVHNodeT<Scalar> &parentNode = nodes[parent];
                    BVHNodeT<Scalar> &leftChild = nodes[parentNode.left_idx];
                    BVHNodeT<Scalar> &rightChild = nodes[parentNode.right_idx];

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
            BVHNodeT<Scalar> &rootNode = nodes[0];
            BVHNodeT<Scalar> &leftChild = nodes[rootNode.left_idx];
            BVHNodeT<Scalar> &rightChild = nodes[rootNode.right_idx];

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

template<typename MortonCodeType = uint64_t, typename Scalar>
std::vector<BVHNodeT<Scalar>>
buildLBVH(tf::Executor &executor,
          const std::vector<PrimitiveT<Scalar>> &primitives,
          MortonSortMethod sortMethod)
{
    if (primitives.empty()) return {};

    std::vector<MortonPrimitive<MortonCodeType>> mortonPrimitives =
            generateMortonCodes<MortonCodeType, Scalar>(primitives, sortMethod, &executor);

    return buildBVH<MortonCodeType, Scalar>(executor, primitives, mortonPrimitives);
}

template<typename MortonCodeType = uint64_t, typename Scalar>
std::vector<BVHNodeT<Scalar>>
buildLBVH(tf::Executor &executor,
          const std::vector<PrimitiveT<Scalar>> &primitives)
{
    return buildLBVH<MortonCodeType, Scalar>(executor, primitives, MortonSortMethod::StdSort);
}

template<typename Scalar, typename Func>
inline void traverseBVH(const std::vector<BVHNodeT<Scalar>> &nodes, Func &&visit)
{
    if (nodes.empty())
    {
        return;
    }

    std::vector<uint32_t> stack;
    stack.reserve(64);
    stack.push_back(0);

    using VisitReturn = std::invoke_result_t<Func &, uint32_t, const BVHNodeT<Scalar> &>;
    constexpr bool returnsBool = std::is_same<VisitReturn, bool>::value;

    while (!stack.empty())
    {
        uint32_t idx = stack.back();
        stack.pop_back();

        const BVHNodeT<Scalar> &node = nodes[idx];

        if constexpr (returnsBool)
        {
            bool traverseChildren = visit(idx, node);
            if (!traverseChildren || node.isLeaf)
            {
                continue;
            }
        }
        else
        {
            visit(idx, node);
            if (node.isLeaf)
            {
                continue;
            }
        }

        stack.push_back(node.right_idx);
        stack.push_back(node.left_idx);
    }
}

template<typename Scalar>
inline bool traverseBVHClosestHit(const std::vector<BVHNodeT<Scalar>> &nodes,
                                  const std::vector<PrimitiveT<Scalar>> &primitives,
                                  const RayT<Scalar> &ray,
                                  uint32_t &outPrimitiveIdx,
                                  Scalar &outT)
{
    if (nodes.empty())
    {
        return false;
    }

    bool hit = false;
    Scalar closestT = ray.tmax;
    uint32_t closestIdx = 0;

    std::vector<uint32_t> stack;
    stack.reserve(64);
    stack.push_back(0);

    while (!stack.empty())
    {
        uint32_t nodeIdx = stack.back();
        stack.pop_back();

        const BVHNodeT<Scalar> &node = nodes[nodeIdx];

        Scalar nodeNear, nodeFar;
        if (!intersectRayAABB<Scalar>(ray, node.bounds, ray.tmin, closestT, nodeNear, nodeFar))
        {
            continue;
        }

        if (node.isLeaf)
        {
            uint32_t primIdx = node.object_idx;
            if (primIdx >= primitives.size())
            {
                continue;
            }

            Scalar tHit;
            if (intersectRayPrimitive<Scalar>(ray, primitives[primIdx], ray.tmin, closestT, tHit))
            {
                if (tHit < closestT)
                {
                    closestT = tHit;
                    closestIdx = primIdx;
                    hit = true;
                }
            }
        }
        else
        {
            const BVHNodeT<Scalar> &left = nodes[node.left_idx];
            const BVHNodeT<Scalar> &right = nodes[node.right_idx];

            Scalar leftNear, leftFar;
            Scalar rightNear, rightFar;
            bool hitLeft = intersectRayAABB<Scalar>(ray, left.bounds, ray.tmin, closestT, leftNear, leftFar);
            bool hitRight = intersectRayAABB<Scalar>(ray, right.bounds, ray.tmin, closestT, rightNear, rightFar);

            if (hitLeft && hitRight)
            {
                if (leftNear < rightNear)
                {
                    stack.push_back(node.right_idx);
                    stack.push_back(node.left_idx);
                }
                else
                {
                    stack.push_back(node.left_idx);
                    stack.push_back(node.right_idx);
                }
            }
            else if (hitLeft)
            {
                stack.push_back(node.left_idx);
            }
            else if (hitRight)
            {
                stack.push_back(node.right_idx);
            }
        }
    }

    if (hit)
    {
        outPrimitiveIdx = closestIdx;
        outT = closestT;
    }

    return hit;
}

#endif //BVH2_BVH_H
