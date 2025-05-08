#ifndef BVH2_OCTREE_H
#define BVH2_OCTREE_H

#include "Csarray.h"
#include "Hilbert.h"

namespace cstone
{
    inline constexpr int digitWeight(int digit)
    {
        int fourGeqmask = -int(digit >= 4);
        return ((7 - digit) & fourGeqmask) - (digit & ~fourGeqmask);
    }

    template<class KeyType>
    inline KeyType encodePlaceholderBit(KeyType key, unsigned prefixLength)
    {
        return key | (KeyType(1) << (sizeof(KeyType) * 8 - 1 - prefixLength));
    }

    template<class KeyType>
    inline KeyType decodePlaceholderBit(KeyType key)
    {
        return key & ~(KeyType(1) << (sizeof(KeyType) * 8 - 1));
    }

    template<class KeyType>
    inline unsigned decodePrefixLength(KeyType key)
    {
        return __builtin_clzll(~key & (KeyType(1) << (sizeof(KeyType) * 8 - 1)));
    }

    template<class KeyType>
    inline unsigned commonPrefix(KeyType a, KeyType b)
    {
        return a == b ? sizeof(KeyType) * 8 : __builtin_clzll(a ^ b);
    }

    template<class KeyType>
    inline constexpr TreeNodeIndex binaryKeyWeight(KeyType key, unsigned level)
    {
        TreeNodeIndex ret = 0;
        for (unsigned l = 1; l <= level + 1; ++l)
        {
            unsigned digit = octalDigit(key, l);
            ret += digitWeight(digit);
        }
        return ret;
    }

    template<class KeyType>
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

    template<class KeyType>
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

    template<class KeyType>
    void getLevelRangeCpu(const KeyType *nodeKeys, TreeNodeIndex numNodes, TreeNodeIndex *levelRange)
    {
        for (unsigned level = 0; level <= maxTreeLevel<KeyType>(); ++level)
        {
            auto it = std::lower_bound(nodeKeys, nodeKeys + numNodes, encodePlaceholderBit(KeyType(0), 3 * level));
            levelRange[level] = TreeNodeIndex(it - nodeKeys);
        }
        levelRange[maxTreeLevel<KeyType>() + 1] = numNodes;
    }

    template<class KeyType>
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

    template<class KeyType>
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

}

#endif //BVH2_OCTREE_H
