#ifndef BVH2_PARALLELRADIXSORT_H
#define BVH2_PARALLELRADIXSORT_H

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>
#include <taskflow/taskflow.hpp>
#include "MortonCode.h"

namespace bvh2
{

template<typename MortonCodeType>
void ChunkedRadixSort(tf::Executor &executor, std::vector<MortonPrimitive<MortonCodeType>> &mortonPrimitives)
{
    const size_t n = mortonPrimitives.size();
    if (n <= 1) return;

    using UnsignedCode = std::make_unsigned_t<MortonCodeType>;

    std::vector<MortonPrimitive<MortonCodeType>> temp(n);

    constexpr int BITS_PER_PASS = 8;
    constexpr int KEY_BITS = std::numeric_limits<UnsignedCode>::digits;
    constexpr int NUM_PASSES = (KEY_BITS + BITS_PER_PASS - 1) / BITS_PER_PASS;
    constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

    std::vector<MortonPrimitive<MortonCodeType>> *src = &mortonPrimitives;
    std::vector<MortonPrimitive<MortonCodeType>> *dst = &temp;

    const size_t maxThreads = std::max<size_t>(1, executor.num_workers());

    for (int pass = 0; pass < NUM_PASSES; pass++)
    {
        const int shift = pass * BITS_PER_PASS;
        const int activeBits = std::min(BITS_PER_PASS, KEY_BITS - shift);
        const UnsignedCode mask = (UnsignedCode(1) << activeBits) - UnsignedCode(1);

        std::vector<std::vector<size_t>> threadHistogram(maxThreads, std::vector<size_t>(NUM_BUCKETS, 0));

        tf::Taskflow taskflow;

        std::vector<tf::Task> hist_tasks;
        hist_tasks.reserve(maxThreads);

        for (size_t t = 0; t < maxThreads; t++)
        {
            hist_tasks.push_back(
                    taskflow.emplace([&, t]() {
                        size_t start = (n * t) / maxThreads;
                        size_t end = (n * (t + 1)) / maxThreads;
                        auto &localHist = threadHistogram[t];
                        for (size_t i = start; i < end; i++)
                        {
                            UnsignedCode code = static_cast<UnsignedCode>((*src)[i].mortonCode);
                            size_t bucket = static_cast<size_t>((code >> shift) & mask);
                            localHist[bucket]++;
                        }
                    })
            );
        }

        std::vector<std::vector<size_t>> threadOffsets(maxThreads, std::vector<size_t>(NUM_BUCKETS, 0));

        auto reduce = taskflow.emplace([&]() {
            size_t prefix = 0;
            for (int b = 0; b < NUM_BUCKETS; ++b)
            {
                size_t sum = 0;

                for (size_t t = 0; t < maxThreads; ++t)
                {
                    threadOffsets[t][b] = sum;
                    sum += threadHistogram[t][b];
                }

                for (size_t t = 0; t < maxThreads; ++t)
                {
                    threadOffsets[t][b] += prefix;
                }
                prefix += sum;
            }
        });

        for (auto &ht: hist_tasks)
        {
            reduce.succeed(ht);
        }

        std::vector<tf::Task> scatter_tasks;
        scatter_tasks.reserve(maxThreads);

        for (size_t t = 0; t < maxThreads; t++)
        {
            scatter_tasks.push_back(
                    taskflow.emplace([&, t]() {
                        size_t start = (n * t) / maxThreads;
                        size_t end = (n * (t + 1)) / maxThreads;
                        auto &localOff = threadOffsets[t];
                        for (size_t i = start; i < end; i++)
                        {
                            UnsignedCode code = static_cast<UnsignedCode>((*src)[i].mortonCode);
                            size_t bucket = static_cast<size_t>((code >> shift) & mask);
                            size_t pos = localOff[bucket]++;
                            (*dst)[pos] = (*src)[i];
                        }
                    })
            );

            scatter_tasks.back().succeed(reduce);
        }

        executor.run(taskflow).wait();

        std::swap(src, dst);
    }

    if (src != &mortonPrimitives)
    {
        mortonPrimitives = temp;
    }
}

} // namespace bvh2

#endif //BVH2_PARALLELRADIXSORT_H
