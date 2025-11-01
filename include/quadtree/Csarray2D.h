#ifndef BVH2_CSARRAY2D_H
#define BVH2_CSARRAY2D_H

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>
#include <limits>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

#include "quadtree/Bitops2D.h"

namespace qtree2d
{
    using TreeNodeIndex = int;
    using LocalIndex = unsigned;

    template<typename Vector>
    std::size_t nNodes(const Vector &tree)
    {
        assert(tree.size());
        return tree.size() - 1;
    }

    template<typename KeyType>
    unsigned calculateNodeCount(KeyType nodeStart, KeyType nodeEnd,
                                const KeyType *codesStart, const KeyType *codesEnd,
                                size_t maxCount)
    {
        auto rangeStart = std::lower_bound(codesStart, codesEnd, nodeStart);
        auto rangeEnd = std::lower_bound(codesStart, codesEnd, nodeEnd);
        size_t count = rangeEnd - rangeStart;
        return static_cast<unsigned>(std::min(count, maxCount));
    }

    template<typename KeyType>
    void computeNodeCounts(const KeyType *tree,
                           unsigned *counts,
                           TreeNodeIndex numNodes,
                           const KeyType *codesStart,
                           const KeyType *codesEnd,
                           unsigned maxCount,
                           tf::Executor &executor)
    {
        tf::Taskflow flow;
        flow.for_each_index(TreeNodeIndex(0), numNodes, TreeNodeIndex(1), [&](TreeNodeIndex i) {
            counts[i] = calculateNodeCount(tree[i], tree[i + 1],
                                           codesStart, codesEnd, maxCount);
        });
        executor.run(flow).wait();
    }

    // return the sibling index (0..3) and level of the specified node
    template<typename KeyType>
    inline std::tuple<int, unsigned> siblingAndLevel(const KeyType *csTree, TreeNodeIndex nodeIdx)
    {
        KeyType thisNode = csTree[nodeIdx];
        KeyType range = csTree[nodeIdx + 1] - thisNode;
        unsigned level = qtTreeLevel<KeyType>(range);

        if (level == 0) { return {-1, level}; }

        int siblingIdx = static_cast<int>(quadDigit<KeyType>(thisNode, level));
        bool siblings = (csTree[nodeIdx - siblingIdx + 4] ==
                         csTree[nodeIdx - siblingIdx] + nodeRange2D<KeyType>(level - 1));
        if (!siblings) { siblingIdx = -1; }

        return {siblingIdx, level};
    }

    // returns 0 for merging, 1 for keep, 4 for splitting
    template<typename KeyType>
    int calculateNodeOp(const KeyType *tree, TreeNodeIndex nodeIdx,
                        const unsigned *counts, unsigned bucketSize)
    {
        auto [siblingIdx, level] = siblingAndLevel<KeyType>(tree, nodeIdx);

        if (siblingIdx > 0)
        {
            auto g = counts + nodeIdx - siblingIdx;
            size_t parentCount = size_t(g[0]) + size_t(g[1]) + size_t(g[2]) + size_t(g[3]);
            bool countMerge = parentCount <= size_t(bucketSize);
            if (countMerge) { return 0; }
        }

        if (counts[nodeIdx] > bucketSize && level < maxTreeLevel2D<KeyType>())
        { return 4; }

        return 1;
    }

    template<typename KeyType, typename LocalIndex>
    bool rebalanceDecision(const KeyType *tree,
                           const unsigned *counts,
                           TreeNodeIndex nNodesIn,
                           unsigned bucketSize,
                           LocalIndex *nodeOps,
                           tf::Executor &executor)
    {
        bool converged = true;

        tf::Taskflow flow;
        std::vector<TreeNodeIndex> indices(nNodesIn);
        std::iota(indices.begin(), indices.end(), 0);

        auto compute = flow.for_each(indices.begin(), indices.end(), [&](TreeNodeIndex i) {
            int decision = calculateNodeOp(tree, i, counts, bucketSize);
            nodeOps[i] = static_cast<LocalIndex>(decision);
        });

        auto reduce = flow.emplace([&]() {
            converged = std::all_of(nodeOps, nodeOps + nNodesIn, [](int v) { return v == 1; });
        });

        compute.precede(reduce);
        executor.run(flow).wait();
        return converged;
    }

    template<typename KeyType>
    void processNode(TreeNodeIndex nodeIndex, const KeyType *oldTree,
                     const TreeNodeIndex *nodeOps, KeyType *newTree)
    {
        KeyType thisNode = oldTree[nodeIndex];
        KeyType range = oldTree[nodeIndex + 1] - thisNode;
        unsigned level = qtTreeLevel<KeyType>(range);

        TreeNodeIndex opCode = nodeOps[nodeIndex + 1] - nodeOps[nodeIndex];
        TreeNodeIndex newNodeIndex = nodeOps[nodeIndex];

        if (opCode == 1)
        {
            newTree[newNodeIndex] = thisNode;
        }
        else if (opCode == 4)
        {
            for (int sibling = 0; sibling < 4; ++sibling)
            {
                newTree[newNodeIndex + sibling] =
                        thisNode + KeyType(sibling) * nodeRange2D<KeyType>(level + 1);
            }
        }
    }

    template<typename InputVector, typename OutputVector>
    void rebalanceTree(const InputVector &tree,
                       OutputVector &newTree,
                       qtree2d::TreeNodeIndex *nodeOps,
                       tf::Executor &executor)
    {
        using KeyType = typename InputVector::value_type;
        TreeNodeIndex numNodesIn = static_cast<TreeNodeIndex>(nNodes(tree));

        std::exclusive_scan(nodeOps, nodeOps + numNodesIn + 1, nodeOps, 0);
        TreeNodeIndex newNumNodes = nodeOps[numNodesIn];

        newTree.resize(static_cast<std::size_t>(newNumNodes) + 1);

        tf::Taskflow flow;
        flow.for_each_index(TreeNodeIndex(0), numNodesIn, TreeNodeIndex(1), [&](TreeNodeIndex i) {
            processNode<KeyType>(i, tree.data(), nodeOps, newTree.data());
        });
        executor.run(flow).wait();
        newTree.back() = tree.back();
    }

    template<typename KeyType>
    bool updateQuadtree(const KeyType *firstKey,
                        const KeyType *lastKey,
                        unsigned bucketSize,
                        std::vector<KeyType> &tree,
                        std::vector<unsigned> &counts,
                        tf::Executor &executor,
                        unsigned maxCount = std::numeric_limits<unsigned>::max())
    {
        std::vector<TreeNodeIndex> nodeOps(nNodes(tree) + 1);
        bool converged = rebalanceDecision<KeyType, TreeNodeIndex>(
                tree.data(), counts.data(), static_cast<TreeNodeIndex>(nNodes(tree)),
                bucketSize, nodeOps.data(), executor);

        std::vector<KeyType> tmpTree;
        rebalanceTree(tree, tmpTree, nodeOps.data(), executor);
        swap(tree, tmpTree);

        counts.resize(nNodes(tree));
        computeNodeCounts(tree.data(), counts.data(), static_cast<TreeNodeIndex>(nNodes(tree)),
                          firstKey, lastKey, maxCount, executor);

        return converged;
    }

    template<typename KeyType = std::uint64_t>
    std::tuple<std::vector<KeyType>, std::vector<unsigned>>
    computeQuadtree(const KeyType *codesStart,
                    const KeyType *codesEnd,
                    unsigned bucketSize,
                    tf::Executor &executor,
                    unsigned maxCount = std::numeric_limits<unsigned>::max())
    {
        std::vector<KeyType> tree{0, nodeRange2D<KeyType>(0)};
        std::vector<unsigned> counts{unsigned(codesEnd - codesStart)};

        while (!updateQuadtree(codesStart, codesEnd, bucketSize, tree, counts, executor, maxCount)) {}

        return std::make_tuple(std::move(tree), std::move(counts));
    }

    template<typename KeyType = std::uint64_t>
    class QuadtreeBuilder
    {
    public:
        explicit QuadtreeBuilder(unsigned bucketSize) : bucketSize_(bucketSize) {}

        std::tuple<std::vector<KeyType>, std::vector<unsigned>> build(
                const KeyType *codesStart, const KeyType *codesEnd,
                tf::Executor &executor,
                unsigned maxCount = std::numeric_limits<unsigned>::max())
        {
            return computeQuadtree<KeyType>(codesStart, codesEnd, bucketSize_, executor, maxCount);
        }

    private:
        unsigned bucketSize_;
    };
}

#endif // BVH2_CSARRAY2D_H

