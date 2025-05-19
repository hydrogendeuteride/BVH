#ifndef BVH2_PARALLELRADIXSORT_H
#define BVH2_PARALLELRADIXSORT_H

#include <vector>
#include <omp.h>
#include <taskflow/taskflow.hpp>
#include "MortonCode.h"

template<typename MortonCodeType>
void ChunkedRadixSort(tf::Executor &executor, std::vector<MortonPrimitive<MortonCodeType>> &mortonPrimitives)
{
    const size_t n = mortonPrimitives.size();
    if (n <= 1) return;

    std::vector<MortonPrimitive<MortonCodeType>> temp(n);

    constexpr int BITS_PER_PASS = 8;
    constexpr int NUM_PASSES = 64 / BITS_PER_PASS;
    constexpr int NUM_BUCKETS = 1 << BITS_PER_PASS;

    std::vector<MortonPrimitive<MortonCodeType>> *src = &mortonPrimitives;
    std::vector<MortonPrimitive<MortonCodeType>> *dst = &temp;

    const int maxThreads = executor.num_workers();

    for (int pass = 0; pass < NUM_PASSES; pass++)
    {
        const int shift = pass * BITS_PER_PASS;
        const uint64_t mask = (static_cast<uint64_t>(NUM_BUCKETS) - 1) << shift;

        std::vector<std::vector<int>> threadHistogram(maxThreads, std::vector<int>(NUM_BUCKETS, 0));

        tf::Taskflow taskflow;

        std::vector<tf::Task> hist_tasks;
        hist_tasks.reserve(maxThreads);

        for (int t = 0; t < maxThreads; t++)
        {
            hist_tasks.push_back(
                    taskflow.emplace([&, t]() {
                        size_t start = (n * t) / maxThreads;
                        size_t end = (n * (t + 1)) / maxThreads;
                        auto &localHist = threadHistogram[t];
                        for (size_t i = start; i < end; i++)
                        {
                            uint64_t code = (*src)[i].mortonCode;
                            uint64_t bucket = (code & mask) >> shift;
                            localHist[bucket]++;
                        }
                    })
            );
        }

        std::vector<int> globalOffsets(NUM_BUCKETS, 0);
        std::vector<std::vector<int>> threadOffsets(maxThreads, std::vector<int>(NUM_BUCKETS, 0));

        auto reduce = taskflow.emplace([&]() {
            uint64_t prefix = 0;
            for (int b = 0; b < NUM_BUCKETS; ++b)
            {
                uint64_t sum = 0;

                for (int t = 0; t < maxThreads; ++t)
                {
                    threadOffsets[t][b] = sum;
                    sum += threadHistogram[t][b];
                }

                globalOffsets[b] = sum;

                for (int t = 0; t < maxThreads; ++t)
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

        for (int t = 0; t < maxThreads; t++)
        {
            scatter_tasks.push_back(
                    taskflow.emplace([&, t]() {
                        size_t start = (n * t) / maxThreads;
                        size_t end = (n * (t + 1)) / maxThreads;
                        auto &localOff = threadOffsets[t];
                        for (size_t i = start; i < end; i++)
                        {
                            uint64_t code = (*src)[i].mortonCode;
                            uint64_t bucket = (code & mask) >> shift;
                            uint64_t pos = localOff[bucket]++;
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

#endif //BVH2_PARALLELRADIXSORT_H
