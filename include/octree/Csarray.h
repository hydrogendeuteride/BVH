#ifndef BVH2_CSARRAY_H
#define BVH2_CSARRAY_H

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>
#include <limits>

#include "util/Bitops.h"

namespace cstone
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
    unsigned calculateNodeCount(
            KeyType nodeStart, KeyType nodeEnd, const KeyType *codesStart, const KeyType *codesEnd, size_t maxCount)
    {
        auto rangeStart = std::lower_bound(codesStart, codesEnd, nodeStart);
        auto rangeEnd = std::lower_bound(codesStart, codesEnd, nodeEnd);
        size_t count = rangeEnd - rangeStart;

        return std::min(count, maxCount);
    }

    /*! count number of particles in each octree node
     *  
     */
    template<typename KeyType>
    void computeNodeCounts(const KeyType *tree,
                           unsigned *counts,
                           TreeNodeIndex numNodes,
                           const KeyType *codesStart,
                           const KeyType *codesEnd,
                           unsigned maxCount)
    {
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < numNodes; ++i)
        {
            counts[i] = calculateNodeCount(tree[i], tree[i + 1], codesStart, codesEnd, maxCount);
        }
    }

    /*! return the sibling index and level of the specified csTree node
     *
     */
    template<typename KeyType>
    inline std::tuple<int, unsigned> siblingAndLevel(const KeyType *csTree, TreeNodeIndex nodeIdx)
    {
        KeyType thisNode = csTree[nodeIdx];
        KeyType range = csTree[nodeIdx + 1] - thisNode;
        unsigned level = treeLevel<KeyType>(range);

        if (level == 0)
        { return {-1, level}; }

        int siblingIdx = octalDigit<KeyType>(thisNode, level);
        bool siblings = (csTree[nodeIdx - siblingIdx + 8] ==
                         csTree[nodeIdx - siblingIdx] + nodeRange<KeyType>(level - 1));
        if (!siblings)
        { siblingIdx = -1; }

        return {siblingIdx, level};
    }

    //! returns 0 for merging, 1 for no-change, 8 for splitting
    template<typename KeyType>
    int calculateNodeOp(const KeyType *tree, TreeNodeIndex nodeIdx, const unsigned *counts, unsigned bucketSize)
    {
        auto [siblingIdx, level] = siblingAndLevel(tree, nodeIdx);

        if (siblingIdx > 0)
        {
            auto g = counts + nodeIdx - siblingIdx;
            size_t parentCount =
                    size_t(g[0]) + size_t(g[1]) + size_t(g[2]) + size_t(g[3]) + size_t(g[4]) + size_t(g[5]) +
                    size_t(g[6]) + size_t(g[7]);
            bool countMerge = parentCount <= size_t(bucketSize);
            if (countMerge)
            { return 0; }
        }

        if (counts[nodeIdx] > bucketSize && level < maxTreeLevel<KeyType>())
        { return 8; }

        return 1;
    }

    /*! Compute split or fuse decision for each octree node in parallel
     *
     */
    template<typename KeyType, typename LocalIndex>
    bool rebalanceDecision(
            const KeyType *tree, const unsigned *counts, TreeNodeIndex nNodes, unsigned bucketSize, LocalIndex *nodeOps)
    {
        bool converged = true;

#pragma omp parallel
        {
            bool convergedThread = true;
#pragma omp for
            for (TreeNodeIndex i = 0; i < nNodes; ++i)
            {
                int decision = calculateNodeOp(tree, i, counts, bucketSize);
                if (decision != 1)
                { convergedThread = false; }

                nodeOps[i] = decision;
            }
            if (!convergedThread)
            { converged = false; }
        }
        return converged;
    }

    /*! transform old nodes into new nodes based on opcodes
     *
     */
    template<typename KeyType>
    void processNode(TreeNodeIndex nodeIndex, const KeyType *oldTree, const TreeNodeIndex *nodeOps, KeyType *newTree)
    {
        KeyType thisNode = oldTree[nodeIndex];
        KeyType range = oldTree[nodeIndex + 1] - thisNode;
        unsigned level = treeLevel<KeyType>(range);

        TreeNodeIndex opCode = nodeOps[nodeIndex + 1] - nodeOps[nodeIndex];
        TreeNodeIndex newNodeIndex = nodeOps[nodeIndex];

        if (opCode == 1)
        { newTree[newNodeIndex] = thisNode; }
        else if (opCode == 8)
        {
            for (int sibling = 0; sibling < 8; ++sibling)
            {
                newTree[newNodeIndex + sibling] = thisNode + sibling * nodeRange<KeyType>(level + 1);
            }
        }
    }

    /*! split or fuse octree nodes based on node counts relative to bucketSize
     *
     */
    template<typename InputVector, typename OutputVector>
    void rebalanceTree(const InputVector &tree, OutputVector &newTree, TreeNodeIndex *nodeOps)
    {
        TreeNodeIndex numNodes = nNodes(tree);

        std::exclusive_scan(nodeOps, nodeOps + numNodes + 1, nodeOps, 0);
        TreeNodeIndex newNumNodes = nodeOps[numNodes];

        newTree.resize(newNumNodes + 1);

#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < numNodes; ++i)
        {
            processNode(i, tree.data(), nodeOps, newTree.data());
        }
        newTree.back() = tree.back();
    }

    /*! update the octree with a single rebalance/count step
     *
     */
    template<typename KeyType>
    bool updateOctree(const KeyType *firstKey,
                      const KeyType *lastKey,
                      unsigned bucketSize,
                      std::vector<KeyType> &tree,
                      std::vector<unsigned> &counts,
                      unsigned maxCount = std::numeric_limits<unsigned>::max())
    {
        std::vector<TreeNodeIndex> nodeOps(nNodes(tree) + 1);
        bool converged = rebalanceDecision(tree.data(), counts.data(), nNodes(tree), bucketSize, nodeOps.data());

        std::vector<KeyType> tmpTree;
        rebalanceTree(tree, tmpTree, nodeOps.data());
        swap(tree, tmpTree);

        counts.resize(nNodes(tree));
        computeNodeCounts(tree.data(), counts.data(), nNodes(tree), firstKey, lastKey, maxCount);

        return converged;
    }

    template<typename KeyType>
    std::tuple<std::vector<KeyType>, std::vector<unsigned>>
    computeOctree(const KeyType *codesStart,
                  const KeyType *codesEnd,
                  unsigned bucketSize,
                  unsigned maxCount = std::numeric_limits<unsigned>::max())
    {
        std::vector<KeyType> tree{0, nodeRange<KeyType>(0)};
        std::vector<unsigned> counts{unsigned(codesEnd - codesStart)};

        while (!updateOctree(codesStart, codesEnd, bucketSize, tree, counts, maxCount));

        return std::make_tuple(std::move(tree), std::move(counts));
    }

    template<typename KeyType = std::uint64_t>
    class OctreeBuilder
    {
    public:
        explicit OctreeBuilder(unsigned bucketSize)
                : bucketSize_(bucketSize)
        {}

        std::tuple<std::vector<KeyType>, std::vector<unsigned>> build(
                const KeyType *codesStart, const KeyType *codesEnd,
                unsigned maxCount = std::numeric_limits<unsigned>::max())
        {
            return computeOctree(codesStart, codesEnd, bucketSize_, maxCount);
        }

    private:
        unsigned bucketSize_;
    };

}

#endif // BVH2_CSARRAY_H