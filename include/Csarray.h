#ifndef BVH2_CSARRAY_H
#define BVH2_CSARRAY_H

#include <cstdint>
#include <vector>
#include <limits>

struct OctreeNode
{
    uint64_t startCode;
    uint64_t endCode;
};

class CornerstoneOctree
{
public:
    using keyType = std::uint64_t;
    using countType = unsigned;
    using indexType = int;

    explicit CornerstoneOctree(unsigned maxParticlePerNode)
            : bucketSize_(maxParticlePerNode)
    {
        codes_ = {0, keyType{1} << 60};
        counts_ = {0};
    }

    void build(const keyType *codesStart, const keyType *codesEnd)
    {
        counts_[0] = static_cast<countType>(codesEnd - codesStart);

        while (!updateOctree(codesStart, codesEnd,
                             std::numeric_limits<countType>::max()));
    }

    const std::vector<keyType> &codes() const
    { return codes_; }

    const std::vector<countType> &counts() const
    { return counts_; }

    std::size_t nNodes() const noexcept
    { return codes_.size() - 1; }

private:
    static inline unsigned treeLevel(keyType range)
    {
        return static_cast<unsigned>(__builtin_ctzll(range) / 3);
    }

    static inline keyType nodeRange(unsigned level)
    {
        return keyType{1} << (3 * level);
    }

    static inline int octalDigit(keyType code, unsigned level)
    {
        return static_cast<int>((code >> (3 * level)) & 0x7ull);
    }

    static countType calculateNodeCount(keyType nodeStart,
                                        keyType nodeEnd,
                                        const keyType *codesStart,
                                        const keyType *codesEnd,
                                        countType maxCount)
    {
        auto *lo = std::lower_bound(codesStart, codesEnd, nodeStart);
        auto *hi = std::lower_bound(codesStart, codesEnd, nodeEnd);
        auto cnt = static_cast<countType>(hi - lo);
        return std::min(cnt, maxCount);
    }

    void computeNodeCounts(const keyType *codesStart,
                           const keyType *codesEnd,
                           countType maxCount)
    {
        const indexType N = static_cast<indexType>(nNodes());
        counts_.resize(N);

#pragma omp parallel for schedule(static)
        for (indexType i = 0; i < N; ++i)
        {
            counts_[i] = calculateNodeCount(codes_[i], codes_[i + 1],
                                            codesStart, codesEnd, maxCount);
        }
    }

    bool rebalanceDecision(std::vector<indexType> &nodeOps)
    {
        bool converged = true;
        const indexType N = static_cast<indexType> (nNodes());

#pragma omp parallel
        {
            bool threadConv = true;
#pragma omp for schedule(static)


        }

        return converged;
    }

    std::vector<keyType> codes_;
    std::vector<countType> counts_;
    unsigned bucketSize_;

};

#endif //BVH2_CSARRAY_H
