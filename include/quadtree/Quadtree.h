#ifndef BVH2_QUADTREE_H
#define BVH2_QUADTREE_H

#include "quadtree/Csarray2D.h"
#include "quadtree/Bitops2D.h"
#include "quadtree/Hilbert2D.h"

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>
#include <algorithm>
#include <vector>

namespace qtree2d
{
    constexpr int digitWeight4(int digit)
    {
        int twoGeqmask = -int(digit >= 2);
        return ((3 - digit) & twoGeqmask) - (digit & ~twoGeqmask);
    }

    template<typename KeyType>
    constexpr TreeNodeIndex binaryKeyWeight4(KeyType key, unsigned level)
    {
        TreeNodeIndex ret = 0;
        for (unsigned l = 1; l <= level + 1; ++l)
        {
            unsigned digit = quadDigit<KeyType>(key, l);
            ret += digitWeight4(digit);
        }
        return ret;
    }

    // Combine internal and leaf nodes into a single array with encoded prefixes (unsorted layout)
    template<typename KeyType>
    void createUnsortedLayout2D(const KeyType *leaves,
                                TreeNodeIndex numInternalNodes,
                                TreeNodeIndex numLeafNodes,
                                KeyType *prefixes,
                                TreeNodeIndex *internalToLeaf,
                                tf::Executor &executor)
    {
        tf::Taskflow flow;
        flow.for_each_index(TreeNodeIndex(0), numLeafNodes, TreeNodeIndex(1), [&](TreeNodeIndex tid) {
            KeyType key = leaves[tid];
            unsigned level = qtTreeLevel<KeyType>(leaves[tid + 1] - key);

            prefixes[tid + numInternalNodes] = encodePlaceholderBit2D<KeyType>(key, 2 * level);
            internalToLeaf[tid + numInternalNodes] = tid + numInternalNodes;

            unsigned prefixLength = commonPrefix2D<KeyType>(key, leaves[tid + 1]);
            if (prefixLength % 2 == 0 && tid < numLeafNodes - 1)
            {
                TreeNodeIndex quadIndex = (tid + binaryKeyWeight4(key, prefixLength / 2)) / 3;
                prefixes[quadIndex] = encodePlaceholderBit2D<KeyType>(key, prefixLength);
                internalToLeaf[quadIndex] = quadIndex;
            }
        });
        executor.run(flow).wait();
    }

    template<typename KeyType>
    void sort_by_key(KeyType *first, KeyType *last, TreeNodeIndex *values)
    {
        size_t n = static_cast<size_t>(last - first);
        std::vector<std::pair<KeyType, TreeNodeIndex>> pairs(n);
        for (size_t i = 0; i < n; ++i) { pairs[i].first = first[i]; pairs[i].second = values[i]; }
        std::sort(pairs.begin(), pairs.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        for (size_t i = 0; i < n; ++i) { first[i] = pairs[i].first; values[i] = pairs[i].second; }
    }

    // Determine the quadtree subdivision level boundaries
    template<typename KeyType>
    void getLevelRange2D(const KeyType *nodeKeys, TreeNodeIndex numNodes, TreeNodeIndex *levelRange)
    {
        for (unsigned level = 0; level <= maxTreeLevel2D<KeyType>(); ++level)
        {
            auto it = std::lower_bound(nodeKeys, nodeKeys + numNodes,
                                       encodePlaceholderBit2D<KeyType>(KeyType(0), 2 * level));
            levelRange[level] = TreeNodeIndex(it - nodeKeys);
        }
        levelRange[maxTreeLevel2D<KeyType>() + 1] = numNodes;
    }

    // Extract parent/child relationships in sorted order
    template<typename KeyType>
    void linkTree2D(const KeyType *prefixes,
                    TreeNodeIndex numInternalNodes,
                    const TreeNodeIndex *leafToInternal,
                    const TreeNodeIndex *levelRange,
                    TreeNodeIndex *childOffsets,
                    TreeNodeIndex *parents,
                    tf::Executor &executor)
    {
        tf::Taskflow flow;
        flow.for_each_index(TreeNodeIndex(0), numInternalNodes, TreeNodeIndex(1), [&](TreeNodeIndex i) {
            TreeNodeIndex idxA = leafToInternal[i];
            KeyType prefix = prefixes[idxA];
            KeyType nodeKey = decodePlaceholderBit2D<KeyType>(prefix);
            unsigned prefixLength = decodePrefixLength2D<KeyType>(prefix);
            unsigned level = prefixLength / 2;

            KeyType childPrefix = encodePlaceholderBit2D<KeyType>(nodeKey, prefixLength + 2);
            TreeNodeIndex leafSearchStart = levelRange[level + 1];
            TreeNodeIndex leafSearchEnd = levelRange[level + 2];

            TreeNodeIndex childIdx =
                    std::lower_bound(prefixes + leafSearchStart,
                                     prefixes + leafSearchEnd,
                                     childPrefix) - prefixes;

            if (childIdx != leafSearchEnd && childPrefix == prefixes[childIdx])
            {
                childOffsets[idxA] = childIdx;
                parents[(childIdx - 1) / 4] = idxA;
            }
        });
        executor.run(flow).wait();
    }

    template<typename T>
    T *rawPtr(std::vector<T> &v) { return v.data(); }
    template<typename T>
    const T *rawPtr(const std::vector<T> &v) { return v.data(); }

    template<typename KeyType>
    struct QuadtreeView
    {
        using NodeType = std::conditional_t<std::is_const_v<KeyType>, const TreeNodeIndex, TreeNodeIndex>;
        TreeNodeIndex numLeafNodes;
        TreeNodeIndex numInternalNodes;
        TreeNodeIndex numNodes;

        KeyType *prefixes;
        NodeType *childOffsets;
        NodeType *parents;
        NodeType *levelRange;
        NodeType *internalToLeaf;
        NodeType *leafToInternal;
    };

    template<typename KeyType>
    class QuadtreeData
    {
    public:
        void resize(TreeNodeIndex numCsLeafNodes)
        {
            numLeafNodes = numCsLeafNodes;
            numInternalNodes = (numLeafNodes - 1) / 3;
            numNodes = numLeafNodes + numInternalNodes;

            prefixes.resize(numNodes);
            internalToLeaf.resize(numNodes);
            leafToInternal.resize(numNodes);
            childOffsets.resize(numNodes + 1);

            TreeNodeIndex parentSize = std::max(1, (numNodes - 1) / 4);
            parents.resize(parentSize);

            levelRange.resize(maxTreeLevel2D<KeyType>() + 2);
        }

        QuadtreeView<KeyType> data()
        {
            return {numLeafNodes, numInternalNodes, numNodes,
                    rawPtr(prefixes), rawPtr(childOffsets), rawPtr(parents),
                    rawPtr(levelRange), rawPtr(internalToLeaf), rawPtr(leafToInternal)};
        }

        QuadtreeView<const KeyType> data() const
        {
            return {numLeafNodes, numInternalNodes, numNodes,
                    rawPtr(prefixes), rawPtr(childOffsets), rawPtr(parents),
                    rawPtr(levelRange), rawPtr(internalToLeaf), rawPtr(leafToInternal)};
        }

        TreeNodeIndex numNodes{0};
        TreeNodeIndex numLeafNodes{0};
        TreeNodeIndex numInternalNodes{0};

        std::vector<KeyType> prefixes;
        std::vector<TreeNodeIndex> childOffsets;
        std::vector<TreeNodeIndex> parents;
        std::vector<TreeNodeIndex> levelRange;
        std::vector<TreeNodeIndex> internalToLeaf;
        std::vector<TreeNodeIndex> leafToInternal;
    };

    template<typename KeyType, typename Func>
    inline void traverseQuadtree(const QuadtreeView<KeyType> &view, Func &&visit)
    {
        if (view.numNodes == 0)
        {
            return;
        }

        using PackedKey = std::remove_cv_t<KeyType>;

        std::vector<TreeNodeIndex> stack;
        stack.reserve(64);
        stack.push_back(0);

        using VisitReturn = std::invoke_result_t<Func &, TreeNodeIndex, PackedKey, unsigned>;
        constexpr bool returnsBool = std::is_same_v<VisitReturn, bool>;

        while (!stack.empty())
        {
            TreeNodeIndex nodeIdx = stack.back();
            stack.pop_back();

            PackedKey packed = static_cast<PackedKey>(view.prefixes[nodeIdx]);
            PackedKey key = decodePlaceholderBit2D<PackedKey>(packed);
            unsigned level = decodePrefixLength2D<PackedKey>(packed) / 2;

            if constexpr (returnsBool)
            {
                if (!visit(nodeIdx, key, level))
                {
                    continue;
                }
            }
            else
            {
                visit(nodeIdx, key, level);
            }

            TreeNodeIndex childStart = view.childOffsets[nodeIdx];
            if (childStart == 0) continue;

            for (int i = 3; i >= 0; --i)
            {
                TreeNodeIndex childIdx = childStart + i;
                if (childIdx >= view.numNodes) continue;
                if (view.parents[(childIdx - 1) / 4] != nodeIdx) continue;
                stack.push_back(childIdx);
            }
        }
    }

    template<typename KeyType>
    void buildLinkedTree(const KeyType *leaves,
                         QuadtreeView<KeyType> q,
                         tf::Executor &executor)
    {
        TreeNodeIndex numNodes = q.numNodes;

        createUnsortedLayout2D(leaves, q.numInternalNodes, q.numLeafNodes,
                               q.prefixes, q.internalToLeaf, executor);

        sort_by_key(q.prefixes, q.prefixes + numNodes, q.internalToLeaf);

        {
            tf::Taskflow flow;
            flow.for_each_index(TreeNodeIndex(0), numNodes, TreeNodeIndex(1), [&](TreeNodeIndex i) {
                q.leafToInternal[q.internalToLeaf[i]] = i;
                q.internalToLeaf[i] -= q.numInternalNodes;
            });
            executor.run(flow).wait();
        }

        getLevelRange2D(q.prefixes, numNodes, q.levelRange);

        std::fill(q.childOffsets, q.childOffsets + numNodes, 0);
        linkTree2D(q.prefixes, q.numInternalNodes,
                   q.leafToInternal, q.levelRange,
                   q.childOffsets, q.parents,
                   executor);
    }

    template<typename KeyType, typename T>
    void nodeFpCenters2D(const KeyType *prefixes,
                         TreeNodeIndex numNodes,
                         Vec2<T> *centers,
                         Vec2<T> *sizes,
                         const Box2D<T> &box,
                         tf::Executor &executor)
    {
        tf::Taskflow flow;
        flow.for_each_index(std::size_t(0), std::size_t(numNodes), std::size_t(1), [&](std::size_t i) {
            KeyType prefix = prefixes[i];
            KeyType startKey = decodePlaceholderBit2D<KeyType>(prefix);
            unsigned level = decodePrefixLength2D<KeyType>(prefix) / 2;

            auto nodeBox = hilbertIBox2D<KeyType>(startKey, level);
            std::tie(centers[i], sizes[i]) = centerAndSize2D<T, KeyType>(nodeBox, box);
        });
        executor.run(flow).wait();
    }

    template<typename KeyType>
    class Quadtree
    {
    public:
        explicit Quadtree(unsigned bucketSize) : bucketSize_(bucketSize) {}

        void build(const KeyType *codesStart,
                   const KeyType *codesEnd,
                   tf::Executor &executor)
        {
            auto [cstree, counts] = computeQuadtree<KeyType>(codesStart, codesEnd,
                                                             bucketSize_, executor);

            data_.resize(static_cast<TreeNodeIndex>(nNodes(cstree)));
            buildLinkedTree(cstree.data(), data_.data(), executor);

            cstoneTree_ = std::move(cstree);
            nodeCounts_ = std::move(counts);
        }

        QuadtreeView<const KeyType> view() const { return data_.data(); }
        const std::vector<KeyType> &cornerstone() const { return cstoneTree_; }
        const std::vector<unsigned> &counts() const { return nodeCounts_; }

    private:
        unsigned bucketSize_;
        QuadtreeData<KeyType> data_;
        std::vector<KeyType> cstoneTree_;
        std::vector<unsigned> nodeCounts_;
    };
}

#endif // BVH2_QUADTREE_H
