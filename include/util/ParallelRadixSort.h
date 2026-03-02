#ifndef BVH2_PARALLELRADIXSORT_H
#define BVH2_PARALLELRADIXSORT_H

#include <algorithm>
#include <array>
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
    using CountType = size_t;

    std::vector<MortonPrimitive<MortonCodeType>> temp(n);

    constexpr size_t BITS_PER_PASS = 8;
    constexpr size_t KEY_BITS = std::numeric_limits<UnsignedCode>::digits;
    constexpr size_t NUM_PASSES = (KEY_BITS + BITS_PER_PASS - 1) / BITS_PER_PASS;
    constexpr size_t NUM_BUCKETS = size_t(1) << BITS_PER_PASS;
    using Histogram = std::array<CountType, NUM_BUCKETS>;

    std::vector<MortonPrimitive<MortonCodeType>> *src = &mortonPrimitives;
    std::vector<MortonPrimitive<MortonCodeType>> *dst = &temp;

    const size_t workerCount = std::max<size_t>(1, executor.num_workers());
    const size_t threadCount = std::min(n, workerCount);

    std::vector<size_t> chunkStarts(threadCount + 1);
    for (size_t t = 0; t <= threadCount; ++t)
    {
        chunkStarts[t] = (n * t) / threadCount;
    }

    std::vector<Histogram> threadHistogram(threadCount);
    std::vector<Histogram> threadOffsets(threadCount);

    size_t shift = 0;
    UnsignedCode mask = 0;

    tf::Taskflow taskflow;

    std::vector<tf::Task> histTasks;
    histTasks.reserve(threadCount);
    for (size_t t = 0; t < threadCount; ++t)
    {
        histTasks.push_back(taskflow.emplace([&, t]() {
            size_t start = chunkStarts[t];
            size_t end = chunkStarts[t + 1];
            auto &localHist = threadHistogram[t];
            localHist.fill(0);

            for (size_t i = start; i < end; ++i)
            {
                UnsignedCode code = static_cast<UnsignedCode>((*src)[i].mortonCode);
                size_t bucket = static_cast<size_t>((code >> shift) & mask);
                ++localHist[bucket];
            }
        }));
    }

    auto reduceTask = taskflow.emplace([&]() {
        CountType bucketPrefix = 0;
        for (size_t b = 0; b < NUM_BUCKETS; ++b)
        {
            CountType threadPrefix = 0;

            for (size_t t = 0; t < threadCount; ++t)
            {
                threadOffsets[t][b] = bucketPrefix + threadPrefix;
                threadPrefix += threadHistogram[t][b];
            }

            bucketPrefix += threadPrefix;
        }
    });

    for (auto &task : histTasks)
    {
        reduceTask.succeed(task);
    }

    std::vector<tf::Task> scatterTasks;
    scatterTasks.reserve(threadCount);
    for (size_t t = 0; t < threadCount; ++t)
    {
        scatterTasks.push_back(taskflow.emplace([&, t]() {
            size_t start = chunkStarts[t];
            size_t end = chunkStarts[t + 1];
            auto &localOff = threadOffsets[t];

            for (size_t i = start; i < end; ++i)
            {
                UnsignedCode code = static_cast<UnsignedCode>((*src)[i].mortonCode);
                size_t bucket = static_cast<size_t>((code >> shift) & mask);
                size_t pos = localOff[bucket]++;
                (*dst)[pos] = (*src)[i];
            }
        }));

        scatterTasks.back().succeed(reduceTask);
    }

    for (size_t pass = 0; pass < NUM_PASSES; ++pass)
    {
        shift = pass * BITS_PER_PASS;
        const size_t activeBits = std::min(BITS_PER_PASS, KEY_BITS - shift);
        mask = (UnsignedCode(1) << activeBits) - UnsignedCode(1);

        executor.run(taskflow).wait();

        std::swap(src, dst);
    }

    if (src != &mortonPrimitives)
    {
        mortonPrimitives.swap(temp);
    }
}

} // namespace bvh2

#endif //BVH2_PARALLELRADIXSORT_H
