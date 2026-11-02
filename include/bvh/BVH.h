#ifndef BVH2_BVH_H
#define BVH2_BVH_H

#include "BVHNode.h"
#include "SAH.h"
#include "util/MortonCode.h"
#include "util/ParallelRadixSort.h"
#include "util/Bitops.h"
#include "util/ray.h"
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <array>
#include <vector>
#include <algorithm>
#include <limits>
#include <type_traits>
#include <utility>

namespace bvh2
{

enum class MortonSortMethod
{
    StdSort,
    RadixSort
};

namespace detail
{

template<typename T, std::size_t Capacity = 64>
class TraversalStack
{
    std::array<T, Capacity> local_;
    std::size_t localSize_ = 0;
    std::vector<T> overflow_;
public:
    bool empty() const { return localSize_ == 0 && overflow_.empty(); }
    void push(const T& value)
    {
        if (localSize_ < Capacity) local_[localSize_++] = value;
        else overflow_.push_back(value);
    }
    T pop()
    {
        if (overflow_.empty()) return local_[--localSize_];
        T value = overflow_.back();
        overflow_.pop_back();
        return value;
    }
};


template<typename MortonCodeType>
inline int commonPrefixWithTieBreak(const std::vector<MortonPrimitive<MortonCodeType>> &mortonPrimitives,
                                    uint32_t first,
                                    uint32_t second)
{
    const MortonPrimitive<MortonCodeType> &a = mortonPrimitives[first];
    const MortonPrimitive<MortonCodeType> &b = mortonPrimitives[second];

    const MortonCodeType codeXor = a.mortonCode ^ b.mortonCode;
    if (codeXor != 0)
    {
        return countLeadingZeros(codeXor);
    }

    const uint32_t primitiveXor = a.primitiveIndex ^ b.primitiveIndex;
    return std::numeric_limits<MortonCodeType>::digits + countLeadingZeros(primitiveXor);
}

template<typename Scalar>
inline void mergeChildBounds(BVHData<Scalar>& nodes, uint32_t idx)
{
    const uint32_t left = nodes.childOffsets[idx], right = left + 1;
    for (int axis = 0; axis < 3; ++axis) {
        nodes.bounds[idx].min[axis] = std::min(nodes.bounds[left].min[axis], nodes.bounds[right].min[axis]);
        nodes.bounds[idx].max[axis] = std::max(nodes.bounds[left].max[axis], nodes.bounds[right].max[axis]);
    }
}

template<typename Scalar>
inline void computeBoundsPostOrder(BVHData<Scalar>& nodes, uint32_t root)
{
    TraversalStack<std::pair<uint32_t, bool>> stack;
    stack.push({root, false});
    while (!stack.empty())
    {
        auto [idx, visited] = stack.pop();
        if (nodes.primitiveIndices[idx] != UINT32_MAX) continue;
        const uint32_t left = nodes.childOffsets[idx], right = left + 1;
        if (!visited) {
            stack.push({idx, true});
            stack.push({right, false});
            stack.push({left, false});
        } else {
            mergeChildBounds(nodes, idx);
        }
    }
}

} // namespace detail

template<typename MortonCodeType>
inline uint32_t
findSplit(const std::vector<MortonPrimitive<MortonCodeType>> &mortonPrimitives, const uint32_t numPrimitives,
          uint32_t first,
          uint32_t last)
{
    (void)numPrimitives;

    if (first == last)
    {
        return first;
    }

    const uint32_t commonPrefix = detail::commonPrefixWithTieBreak(mortonPrimitives, first, last);

    uint32_t split = first;
    uint32_t step = last - first;

    do
    {
        step = (step + 1) >> 1;
        uint32_t newSplit = split + step;

        if (newSplit < last)
        {
            const uint32_t splitPrefix = detail::commonPrefixWithTieBreak(mortonPrimitives, first, newSplit);

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
inline int commonUpperBits(const std::vector<MortonPrimitive<MortonCodeType>> &mortonPrimitives,
                           int first,
                           int second)
{
    if (second < 0 || second >= static_cast<int>(mortonPrimitives.size()))
    {
        return -1;
    }

    return detail::commonPrefixWithTieBreak(
            mortonPrimitives,
            static_cast<uint32_t>(first),
            static_cast<uint32_t>(second));
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

    const int index = static_cast<int>(idx);
    const int L_delta = (idx > 0) ? commonUpperBits(mortonPrimitives, index, index - 1) : -1;

    const int R_delta = (idx < numPrimitives - 1) ? commonUpperBits(mortonPrimitives, index, index + 1) : -1;

    const int d = (R_delta > L_delta) ? 1 : -1;

    const int delta_min = std::min(L_delta, R_delta);
    int l_max = 2;
    int delta = -1;
    int i_tmp = index + d * l_max;
    delta = commonUpperBits(mortonPrimitives, index, i_tmp);

    while (delta > delta_min)
    {
        l_max <<= 1;
        i_tmp = index + d * l_max;
        delta = commonUpperBits(mortonPrimitives, index, i_tmp);
    }

    int l = 0;
    int t = l_max >> 1;
    while (t > 0)
    {
        i_tmp = index + (l + t) * d;
        delta = commonUpperBits(mortonPrimitives, index, i_tmp);
        if (delta > delta_min)
        {
            l += t;
        }
        t >>= 1;
    }

    int jdx = index + l * d;
    uint32_t first = static_cast<uint32_t>(index);
    uint32_t second = static_cast<uint32_t>(jdx);
    if (d < 0)
    {
        std::swap(first, second);
    }

    return {first, second};
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
                      if (a.mortonCode != b.mortonCode)
                      {
                          return a.mortonCode < b.mortonCode;
                      }

                      return a.primitiveIndex < b.primitiveIndex;
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
BVHData<Scalar>
buildBVH(tf::Executor &executor, const std::vector<PrimitiveT<Scalar>> &primitives,
         const std::vector<MortonPrimitive<MortonCodeType>> &mortonPrimitives)
{
    if (primitives.empty()) return {};
    if (primitives.size() > UINT32_MAX / 2)
        throw std::runtime_error("BVH node indices exceed uint32 capacity");

    const uint32_t numPrimitives = static_cast<uint32_t>(primitives.size());
    BVHData<Scalar> nodes;
    nodes.resize(2 * numPrimitives - 1);
    if (numPrimitives == 1)
    {
        nodes.primitiveIndices[0] = mortonPrimitives[0].primitiveIndex;
        nodes.bounds[0] = primitives[nodes.primitiveIndices[0]].bounds;
        return nodes;
    }

    const uint32_t numInternalNodes = numPrimitives - 1;
    nodes.childOffsets[0] = 1;
    tf::Taskflow tf;
    auto taskNodes = tf.for_each_index(0u, numInternalNodes, 1u, [&](uint32_t idx) {
        const PrimitiveRange range = determineRange(idx, numPrimitives, mortonPrimitives);
        const uint32_t gamma = findSplit(mortonPrimitives, numPrimitives, range.first, range.last);
        const uint32_t left = gamma + (range.first == gamma ? numInternalNodes : 0);
        const uint32_t right = gamma + 1 + (range.last == gamma + 1 ? numInternalNodes : 0);

        // Karras internal node idx owns child slots 2*idx+1 and 2*idx+2.
        // Emit those children directly, without a second node array or remapping pass.
        auto emit = [&](uint32_t target, uint32_t source) {
            if (source < numInternalNodes) nodes.childOffsets[target] = 2 * source + 1;
            else {
                nodes.primitiveIndices[target] = mortonPrimitives[source - numInternalNodes].primitiveIndex;
                nodes.bounds[target] = primitives[nodes.primitiveIndices[target]].bounds;
            }
        };
        emit(2 * idx + 1, left);
        emit(2 * idx + 2, right);
    });
    auto taskBounds = tf.emplace([&](tf::Subflow& sf) {
        if (executor.num_workers() == 1 || numPrimitives < 4096) {
            detail::computeBoundsPostOrder(nodes, 0);
            return;
        }
        // Disjoint subtrees need no atomics; merge their ancestors after joining.
        std::vector<uint32_t> frontier{0};
        size_t first = 0;
        while (first < frontier.size() && frontier.size() - first < 4 * executor.num_workers()) {
            const uint32_t idx = frontier[first++];
            if (nodes.primitiveIndices[idx] != UINT32_MAX) continue;
            const uint32_t left = nodes.childOffsets[idx];
            frontier.push_back(left);
            frontier.push_back(left + 1);
        }
        for (size_t i = first; i < frontier.size(); ++i) {
            const uint32_t root = frontier[i];
            sf.emplace([&, root]() { detail::computeBoundsPostOrder(nodes, root); });
        }
        sf.join();
        while (first) {
            const uint32_t idx = frontier[--first];
            if (nodes.primitiveIndices[idx] == UINT32_MAX) detail::mergeChildBounds(nodes, idx);
        }
    });
    taskNodes.precede(taskBounds);
    executor.run(tf).get();

    return nodes;
}

template<typename MortonCodeType = uint64_t, typename Scalar>
BVHData<Scalar>
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
BVHData<Scalar>
buildLBVH(tf::Executor &executor,
          const std::vector<PrimitiveT<Scalar>> &primitives)
{
    return buildLBVH<MortonCodeType, Scalar>(executor, primitives, MortonSortMethod::StdSort);
}

template<typename Scalar, typename Func>
inline void traverseBVH(const BVHView<Scalar> &nodes, Func &&visit)
{
    using S = std::remove_const_t<Scalar>;
    if (nodes.empty())
    {
        return;
    }

    detail::TraversalStack<uint32_t> stack;
    stack.push(0);

    using VisitReturn = std::invoke_result_t<Func &, uint32_t, const BVHNodeT<S> &>;
    constexpr bool returnsBool = std::is_same<VisitReturn, bool>::value;

    while (!stack.empty())
    {
        const uint32_t idx = stack.pop();

        const BVHNodeT<S> &node = nodes[idx];

        if constexpr (returnsBool)
        {
            bool traverseChildren = visit(idx, node);
            if (!traverseChildren || node.isLeaf())
            {
                continue;
            }
        }
        else
        {
            visit(idx, node);
            if (node.isLeaf())
            {
                continue;
            }
        }

        stack.push(node.right_idx());
        stack.push(node.left_idx);
    }
}

template<typename Scalar>
inline bool traverseBVHClosestHit(const BVHView<Scalar> &nodes,
                                  const std::vector<PrimitiveT<std::remove_const_t<Scalar>>> &primitives,
                                  const RayT<std::remove_const_t<Scalar>> &ray,
                                  uint32_t &outPrimitiveIdx,
                                  std::remove_const_t<Scalar> &outT)
{
    using S = std::remove_const_t<Scalar>;
    if (nodes.empty())
    {
        return false;
    }

    bool hit = false;
    S closestT = ray.tmax;
    uint32_t closestIdx = 0;

    const detail::PreparedRayT<S> prepared(ray);
    S rootNear, rootFar;
    if (!detail::intersectPreparedRayAABB(prepared, nodes.bounds[0], ray.tmin, closestT, rootNear, rootFar))
        return false;
    struct StackEntry { uint32_t nodeIdx; S nearT; };
    detail::TraversalStack<StackEntry> stack;
    uint32_t nodeIdx = 0;

    while (true)
    {
        const uint32_t primitiveIndex = nodes.primitiveIndices[nodeIdx];

        if (primitiveIndex != UINT32_MAX)
        {
            const uint32_t primIdx = primitiveIndex;
            S tHit;
            if (primIdx < primitives.size() &&
                intersectRayPrimitive<S>(ray, primitives[primIdx], ray.tmin, closestT, tHit) &&
                (!hit || tHit < closestT))
            {
                closestT = tHit;
                closestIdx = primIdx;
                hit = true;
            }
        }
        else
        {
            const uint32_t left = nodes.childOffsets[nodeIdx], right = left + 1;

            S leftNear, rightNear;
            const unsigned hitMask = detail::intersectPreparedRayAABBPair(
                prepared, nodes.bounds + left, ray.tmin, closestT, leftNear, rightNear);
            const bool hitLeft = hitMask & 1, hitRight = hitMask & 2;

            if (hitLeft && hitRight)
            {
                if (leftNear < rightNear)
                {
                    stack.push({right, rightNear});
                    nodeIdx = left;
                }
                else
                {
                    stack.push({left, leftNear});
                    nodeIdx = right;
                }
                continue;
            }
            else if (hitLeft)
            {
                nodeIdx = left;
                continue;
            }
            else if (hitRight)
            {
                nodeIdx = right;
                continue;
            }
        }
        if (stack.empty()) break;
        StackEntry next = stack.pop();
        while (next.nearT > closestT && !stack.empty()) next = stack.pop();
        if (next.nearT > closestT) break;
        nodeIdx = next.nodeIdx;
    }

    if (hit)
    {
        outPrimitiveIdx = closestIdx;
        outT = closestT;
    }

    return hit;
}

template<typename Scalar>
inline bool traverseBVHClosestNodeHit(const BVHView<Scalar> &nodes,
                                      const RayT<std::remove_const_t<Scalar>> &ray,
                                      uint32_t maxDepth,
                                      uint32_t &outNodeIdx,
                                      std::remove_const_t<Scalar> &outT)
{
    using S = std::remove_const_t<Scalar>;
    if (nodes.empty())
    {
        return false;
    }

    struct StackEntry
    {
        uint32_t nodeIdx;
        uint32_t depth;
        S nearT;
    };

    bool hit = false;
    S closestT = ray.tmax;
    uint32_t closestIdx = 0;

    const detail::PreparedRayT<S> prepared(ray);
    S rootNear, rootFar;
    if (!detail::intersectPreparedRayAABB(prepared, nodes.bounds[0], ray.tmin, closestT, rootNear, rootFar))
        return false;
    detail::TraversalStack<StackEntry> stack;
    stack.push({0, 0, rootNear});

    while (!stack.empty())
    {
        const StackEntry entry = stack.pop();
        if (entry.nearT > closestT) continue;
        const uint32_t idx = entry.nodeIdx;
        const S nodeNear = entry.nearT;

        if (nodes.isLeaf(idx) || entry.depth >= maxDepth)
        {
            if (!hit || nodeNear < closestT)
            {
                closestT = nodeNear;
                closestIdx = entry.nodeIdx;
                hit = true;
            }
            continue;
        }

        const uint32_t left = nodes.childOffsets[idx], right = left + 1;

        S leftNear, rightNear;
        const unsigned hitMask = detail::intersectPreparedRayAABBPair(
            prepared, nodes.bounds + left, ray.tmin, closestT, leftNear, rightNear);
        const bool hitLeft = hitMask & 1, hitRight = hitMask & 2;

        if (hitLeft && hitRight)
        {
            if (leftNear < rightNear)
            {
                stack.push({right, entry.depth + 1, rightNear});
                stack.push({left, entry.depth + 1, leftNear});
            }
            else
            {
                stack.push({left, entry.depth + 1, leftNear});
                stack.push({right, entry.depth + 1, rightNear});
            }
        }
        else if (hitLeft)
        {
            stack.push({left, entry.depth + 1, leftNear});
        }
        else if (hitRight)
        {
            stack.push({right, entry.depth + 1, rightNear});
        }
    }

    if (hit)
    {
        outNodeIdx = closestIdx;
        outT = closestT;
    }

    return hit;
}

template<typename Scalar, typename Func>
inline void traverseBVH(const BVHData<Scalar>& nodes, Func&& visit) {
    traverseBVH(nodes.data(), std::forward<Func>(visit));
}

template<typename Scalar>
inline bool traverseBVHClosestHit(const BVHData<Scalar>& nodes,
                                  const std::vector<PrimitiveT<Scalar>>& primitives,
                                  const RayT<Scalar>& ray, uint32_t& id, Scalar& t) {
    return traverseBVHClosestHit(nodes.data(), primitives, ray, id, t);
}

template<typename Scalar>
inline bool traverseBVHClosestNodeHit(const BVHData<Scalar>& nodes, const RayT<Scalar>& ray,
                                      uint32_t maxDepth, uint32_t& id, Scalar& t) {
    return traverseBVHClosestNodeHit(nodes.data(), ray, maxDepth, id, t);
}

} // namespace bvh2

#endif //BVH2_BVH_H
