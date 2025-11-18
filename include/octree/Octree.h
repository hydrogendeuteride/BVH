#ifndef BVH2_OCTREE_H
#define BVH2_OCTREE_H

#include "Csarray.h"
#include "util/Hilbert.h"
#include "util/Bitops.h"

#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

namespace cstone
{
    constexpr int digitWeight(int digit)
    {
        int fourGeqmask = -int(digit >= 4);
        return ((7 - digit) & fourGeqmask) - (digit & ~fourGeqmask);
    }

    /*! map a binary node index to an octree node index
     *
     */
    template<typename KeyType>
    constexpr TreeNodeIndex binaryKeyWeight(KeyType key, unsigned level)
    {
        TreeNodeIndex ret = 0;
        for (unsigned l = 1; l <= level + 1; ++l)
        {
            unsigned digit = octalDigit(key, l);
            ret += digitWeight(digit);
        }
        return ret;
    }

    /*! combine internal and leaf tree parts into a single array with the nodeKey prefixes
     *
     *  prefixes: output octree SFC keys, length @p numInternalNodes + numLeafNodes
     */
    template<typename KeyType>
    void createUnsortedLayoutCpu(const KeyType *leaves,
                                 TreeNodeIndex numInternalNodes,
                                 TreeNodeIndex numLeafNodes,
                                 KeyType *prefixes,
                                 TreeNodeIndex *internalToLeaf,
                                 tf::Executor &executor)
    {
        tf::Taskflow flow;
        flow.for_each_index(TreeNodeIndex(0), numLeafNodes, TreeNodeIndex(1),
                            [&](TreeNodeIndex tid) {
                                KeyType key = leaves[tid];
                                unsigned level = treeLevel(leaves[tid + 1] - key);

                                prefixes[tid + numInternalNodes] = encodePlaceholderBit(key, 3 * level);
                                internalToLeaf[tid + numInternalNodes] = tid + numInternalNodes;

                                unsigned prefixLength = commonPrefix(key, leaves[tid + 1]);
                                if (prefixLength % 3 == 0 && tid < numLeafNodes - 1)
                                {
                                    TreeNodeIndex octIndex = (tid + binaryKeyWeight(key, prefixLength / 3)) / 7;
                                    prefixes[octIndex] = encodePlaceholderBit(key, prefixLength);
                                    internalToLeaf[octIndex] = octIndex;
                                }
                            });
        executor.run(flow).wait();
    }

    /*! extract parent/child relationships from binary tree and translate to sorted order
     *
     */
    template<typename KeyType>
    void linkTreeCpu(const KeyType *prefixes,
                     TreeNodeIndex numInternalNodes,
                     const TreeNodeIndex *leafToInternal,
                     const TreeNodeIndex *levelRange,
                     TreeNodeIndex *childOffsets,
                     TreeNodeIndex *parents,
                     tf::Executor &executor)
    {
        tf::Taskflow flow;
        flow.for_each_index(TreeNodeIndex(0), numInternalNodes, TreeNodeIndex(1),
                            [&](TreeNodeIndex i) {
                                TreeNodeIndex idxA = leafToInternal[i];
                                KeyType prefix = prefixes[idxA];
                                KeyType nodeKey = decodePlaceholderBit(prefix);
                                unsigned prefixLength = decodePrefixLength(prefix);
                                unsigned level = prefixLength / 3;

                                KeyType childPrefix = encodePlaceholderBit(nodeKey, prefixLength + 3);
                                TreeNodeIndex leafSearchStart = levelRange[level + 1];
                                TreeNodeIndex leafSearchEnd = levelRange[level + 2];

                                TreeNodeIndex childIdx =
                                        std::lower_bound(prefixes + leafSearchStart,
                                                         prefixes + leafSearchEnd,
                                                         childPrefix) - prefixes;

                                if (childIdx != leafSearchEnd && childPrefix == prefixes[childIdx])
                                {
                                    childOffsets[idxA] = childIdx;
                                    parents[(childIdx - 1) / 8] = idxA;
                                }
                            });
        executor.run(flow).wait();
    }

    //! determine the octree subdivision level boundaries
    template<typename KeyType>
    void getLevelRangeCpu(const KeyType *nodeKeys, TreeNodeIndex numNodes, TreeNodeIndex *levelRange)
    {
        for (unsigned level = 0; level <= maxTreeLevel<KeyType>(); ++level)
        {
            auto it = std::lower_bound(nodeKeys, nodeKeys + numNodes, encodePlaceholderBit(KeyType(0), 3 * level));
            levelRange[level] = TreeNodeIndex(it - nodeKeys);
        }
        levelRange[maxTreeLevel<KeyType>() + 1] = numNodes;
    }

    template<typename KeyType>
    void sort_by_key(KeyType *first, KeyType *last, TreeNodeIndex *values)
    {
        size_t n = last - first;
        std::vector<std::pair<KeyType, TreeNodeIndex>> pairs(n);

        for (size_t i = 0; i < n; ++i)
        {
            pairs[i].first = first[i];
            pairs[i].second = values[i];
        }

        std::sort(pairs.begin(), pairs.end(),
                  [](const auto &a, const auto &b) { return a.first < b.first; });

        for (size_t i = 0; i < n; ++i)
        {
            first[i] = pairs[i].first;
            values[i] = pairs[i].second;
        }
    }

    template<typename KeyType>
    void buildOctreeCpu(const KeyType *cstoneTree,
                        TreeNodeIndex numLeafNodes,
                        TreeNodeIndex numInternalNodes,
                        KeyType *prefixes,
                        TreeNodeIndex *childOffsets,
                        TreeNodeIndex *parents,
                        TreeNodeIndex *levelRange,
                        TreeNodeIndex *internalToLeaf,
                        TreeNodeIndex *leafToInternal,
                        tf::Executor &executor)
    {
        TreeNodeIndex numNodes = numInternalNodes + numLeafNodes;

        createUnsortedLayoutCpu(cstoneTree,
                                numInternalNodes,
                                numLeafNodes,
                                prefixes,
                                internalToLeaf,
                                executor);

        sort_by_key(prefixes, prefixes + numNodes, internalToLeaf);

        {
            tf::Taskflow flow;
            flow.for_each_index(TreeNodeIndex(0), numNodes, TreeNodeIndex(1),
                                [&](TreeNodeIndex i) {
                                    leafToInternal[internalToLeaf[i]] = i;
                                    internalToLeaf[i] -= numInternalNodes;
                                });
            executor.run(flow).wait();
        }

        getLevelRangeCpu(prefixes, numNodes, levelRange);

        std::fill(childOffsets, childOffsets + numNodes, 0);
        linkTreeCpu(prefixes, numInternalNodes,
                    leafToInternal, levelRange,
                    childOffsets, parents,
                    executor);
    }

    template<typename KeyType>
    struct OctreeView
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

    template<typename T, class KeyType>
    struct OctreeNsView
    {
        const Vec3<T> *centers;
        const Vec3<T> *sizes;

        const TreeNodeIndex *childOffsets;
        const TreeNodeIndex *internalToLeaf;

        const LocalIndex *layout;
    };

    template<typename KeyType, typename Func>
    inline void traverseOctree(const OctreeView<KeyType> &view, Func &&visit)
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
            PackedKey key = decodePlaceholderBit<PackedKey>(packed);
            unsigned level = decodePrefixLength<PackedKey>(packed) / 3;

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

            for (int i = 7; i >= 0; --i)
            {
                TreeNodeIndex childIdx = childStart + i;
                if (childIdx >= view.numNodes) continue;
                if (view.parents[(childIdx - 1) / 8] != nodeIdx) continue;
                stack.push_back(childIdx);
            }
        }
    }

    template<typename T>
    T *rawPtr(std::vector<T> &v)
    {
        return v.data();
    }

    template<typename T>
    const T *rawPtr(const std::vector<T> &v)
    {
        return v.data();
    }

    template<typename KeyType>
    class OctreeData
    {
    public:
        void resize(TreeNodeIndex numCsLeafNodes)
        {
            numLeafNodes = numCsLeafNodes;
            numInternalNodes = (numLeafNodes - 1) / 7;
            numNodes = numLeafNodes + numInternalNodes;

            prefixes.resize(numNodes);
            internalToLeaf.resize(numNodes);
            leafToInternal.resize(numNodes);
            childOffsets.resize(numNodes + 1);

            TreeNodeIndex parentSize = std::max(1, (numNodes - 1) / 8);
            parents.resize(parentSize);

            levelRange.resize(maxTreeLevel<KeyType>() + 2);
        }

        OctreeView<KeyType> data()
        {
            return {numLeafNodes, numInternalNodes, numNodes,
                    rawPtr(prefixes), rawPtr(childOffsets), rawPtr(parents),
                    rawPtr(levelRange), rawPtr(internalToLeaf), rawPtr(leafToInternal)};
        }

        OctreeView<const KeyType> data() const
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

    template<typename KeyType>
    void buildLinkedTree(const KeyType *leaves,
                         OctreeView<KeyType> o,
                         tf::Executor &executor)
    {
        buildOctreeCpu(leaves,
                       o.numLeafNodes,
                       o.numInternalNodes,
                       o.prefixes,
                       o.childOffsets,
                       o.parents,
                       o.levelRange,
                       o.internalToLeaf,
                       o.leafToInternal,
                       executor);
    }

    template<typename KeyType, typename T>
    void nodeFpCenters(const KeyType *prefixes,
                       TreeNodeIndex numNodes,
                       Vec3<T> *centers,
                       Vec3<T> *sizes,
                       const Box<T> &box,
                       tf::Executor &executor)
    {
        tf::Taskflow flow;
        flow.for_each_index(std::size_t(0), std::size_t(numNodes), std::size_t(1),
                            [&](std::size_t i) {
                                KeyType prefix = prefixes[i];
                                KeyType startKey = decodePlaceholderBit(prefix);
                                unsigned level = decodePrefixLength(prefix) / 3;

                                auto nodeBox = hilbertIBox(startKey, level);
                                std::tie(centers[i], sizes[i]) = centerAndSize(nodeBox, box);
                            });
        executor.run(flow).wait();
    }

    template<typename KeyType>
    class Octree
    {
    public:
        explicit Octree(unsigned bucketSize)
                : bucketSize_(bucketSize)
        {
        }

        void build(const KeyType *codesStart,
                   const KeyType *codesEnd,
                   tf::Executor &executor)
        {
            auto [cstree, counts] = computeOctree(codesStart, codesEnd,
                                                  bucketSize_, executor);

            octreeData_.resize(nNodes(cstree));
            buildLinkedTree(cstree.data(), octreeData_.data(), executor);

            cstoneTree_ = std::move(cstree);
            nodeCounts_ = std::move(counts);
        }

        OctreeView<const KeyType> view() const
        {
            return octreeData_.data();
        }

        const std::vector<KeyType> &cornerstone() const
        {
            return cstoneTree_;
        }

        const std::vector<unsigned> &counts() const
        {
            return nodeCounts_;
        }

    private:
        unsigned bucketSize_;
        OctreeData<KeyType> octreeData_;
        std::vector<KeyType> cstoneTree_;
        std::vector<unsigned> nodeCounts_;
    };
}

#endif //BVH2_OCTREE_H
