#ifndef BVH2_OCTREE_H
#define BVH2_OCTREE_H

#include "Csarray.h"
#include "Hilbert.h"

#include "util/Bitops.h"

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
                                 TreeNodeIndex *internalToLeaf)
    {
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex tid = 0; tid < numLeafNodes; ++tid)
        {
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
        }
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
                     TreeNodeIndex *parents)
    {
#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < numInternalNodes; ++i)
        {
            TreeNodeIndex idxA = leafToInternal[i];
            KeyType prefix = prefixes[idxA];
            KeyType nodeKey = decodePlaceholderBit(prefix);
            unsigned prefixLength = decodePrefixLength(prefix);
            unsigned level = prefixLength / 3;
            assert(level < maxTreeLevel<KeyType>());

            KeyType childPrefix = encodePlaceholderBit(nodeKey, prefixLength + 3);

            TreeNodeIndex leafSearchStart = levelRange[level + 1];
            TreeNodeIndex leafSearchEnd = levelRange[level + 2];
            TreeNodeIndex childIdx =
                    std::lower_bound(prefixes + leafSearchStart, prefixes + leafSearchEnd, childPrefix) - prefixes;

            if (childIdx != leafSearchEnd && childPrefix == prefixes[childIdx])
            {
                childOffsets[idxA] = childIdx;
                parents[(childIdx - 1) / 8] = idxA;
            }
        }
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
                        TreeNodeIndex *leafToInternal)
    {
        TreeNodeIndex numNodes = numInternalNodes + numLeafNodes;
        createUnsortedLayoutCpu(cstoneTree, numInternalNodes, numLeafNodes, prefixes, internalToLeaf);
        sort_by_key(prefixes, prefixes + numNodes, internalToLeaf);

#pragma omp parallel for schedule(static)
        for (TreeNodeIndex i = 0; i < numNodes; ++i)
        {
            leafToInternal[internalToLeaf[i]] = i;
            internalToLeaf[i] -= numInternalNodes;
        }
        getLevelRangeCpu(prefixes, numNodes, levelRange);

        std::fill(childOffsets, childOffsets + numNodes, 0);
        linkTreeCpu(prefixes, numInternalNodes, leafToInternal, levelRange, childOffsets, parents);
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
    void buildLinkedTree(const KeyType *leaves, OctreeView<KeyType> o)
    {
        buildOctreeCpu(leaves, o.numLeafNodes, o.numInternalNodes, o.prefixes, o.childOffsets, o.parents, o.levelRange,
                       o.internalToLeaf, o.leafToInternal);
    }

    template<typename KeyType, class T>
    void
    nodeFpCenters(const KeyType *prefixes, TreeNodeIndex numNodes, Vec3<T> *centers, Vec3<T> *sizes, const Box<T> &box)
    {
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < numNodes; ++i)
        {
            KeyType prefix = prefixes[i];
            KeyType startKey = decodePlaceholderBit(prefix);
            unsigned level = decodePrefixLength(prefix) / 3;
            auto nodeBox = hilbertIBox(startKey, level);
            std::tie(centers[i], sizes[i]) = centerAndSize<KeyType>(nodeBox, box);
        }
    }

    template<typename KeyType = std::uint64_t>
    class Octree
    {
    public:
        explicit Octree(unsigned bucketSize)
                : bucketSize_(bucketSize)
        {
        }

        void build(const KeyType *codesStart, const KeyType *codesEnd)
        {
            auto [cstree, counts] = computeOctree(codesStart, codesEnd, bucketSize_);

            octreeData_.resize(nNodes(cstree));

            buildLinkedTree(cstree.data(), octreeData_.data());

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
