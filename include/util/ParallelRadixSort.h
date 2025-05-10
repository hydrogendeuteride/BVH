#ifndef BVH2_PARALLELRADIXSORT_H
#define BVH2_PARALLELRADIXSORT_H

#include <vector>
#include <omp.h>
#include "MortonCode.h"

void ChunkedRadixSort(std::vector<MortonPrimitive> &mortonPrimitives)
{
    const size_t n = mortonPrimitives.size();
    if (n <= 1) return;

    std::vector<MortonPrimitive> temp(n);

    constexpr int BITS_PER_PASS = 8 ;
    constexpr int NUM_PASSES = 64 / BITS_PER_PASS;
    constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

    std::vector<MortonPrimitive> *src = &mortonPrimitives;
    std::vector<MortonPrimitive> *dst = &temp;

    int maxThreads = omp_get_max_threads();

    for (int pass = 0; pass < NUM_PASSES; pass++)
    {
        const int shift = pass * BITS_PER_PASS;
        const uint64_t mask = (static_cast<uint64_t>(NUM_BUCKETS) - 1) << shift;

        std::vector<std::vector<int>> threadHistogram(maxThreads, std::vector<int>(NUM_BUCKETS, 0));

#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &localHist = threadHistogram[tid];

#pragma omp for nowait
            for (size_t i = 0; i < n; ++i)
            {
                uint64_t code = (*src)[i].mortonCode;
                uint64_t bucket = (code & mask) >> shift;

                if (bucket < NUM_BUCKETS)
                {
                    ++localHist[bucket];
                }
                else
                {
                    bucket = NUM_BUCKETS - 1;
                    ++localHist[bucket];
                }
            }
        }

        std::vector<int> globalOffsets(NUM_BUCKETS, 0);
        std::vector<std::vector<int>> threadOffsets(maxThreads, std::vector<int>(NUM_BUCKETS, 0));

        for (int b = 0; b < NUM_BUCKETS; b++)
        {
            uint64_t sum = 0;
            for (int t = 0; t < maxThreads; ++t)
            {
                threadOffsets[t][b] = sum;
                sum += threadHistogram[t][b];
            }
            globalOffsets[b] = sum;
        }

        uint64_t prefixSum = 0;
        for (int b = 0; b < NUM_BUCKETS; ++b)
        {
            uint64_t count = globalOffsets[b];
            globalOffsets[b] += prefixSum;

            for (int t = 0; t < maxThreads; ++t)
            {
                threadOffsets[t][b] += prefixSum;
            }

            prefixSum += count;
        }

#pragma omp parallel
        {
            uint64_t tid = omp_get_thread_num();
            auto &localOffsets = threadOffsets[tid];

#pragma omp for nowait
            for (int i = 0; i < n; ++i)
            {
                uint64_t code = (*src)[i].mortonCode;
                uint64_t bucket = (code & mask) >> shift;
                if (bucket < NUM_BUCKETS)
                {
                    uint64_t pos = localOffsets[bucket]++;
                    (*dst)[pos] = (*src)[i];
                }
                else
                {
                    bucket = NUM_BUCKETS - 1;
                    uint64_t pos = localOffsets[bucket]++;
                    (*dst)[pos] = (*src)[i];
                }
            }
        }

        std::swap(src, dst);
    }

    if (src != &mortonPrimitives)
    {
        mortonPrimitives = temp;
    }
}

#endif //BVH2_PARALLELRADIXSORT_H
