# BVH

Fast Linear Bounding Volume Hierarchy generation C++ header-only library with taskflow multithreading.

## Features

- LBVH (Linear Bounding Volume Hierarchy)
- Octree
- Quadtree (WIP)

## Installation

BVH2 is a header-only library. Simply include the headers in your project:
```c++
#include "bvh/BVH.h"
#include "octree/Octree.h"
#include "util/ParallelRadixSort.h"
```

## Usage

Parallel Radix Sort
```c++
#include "util/ParallelRadixSort.h"
#include <vector>

int main() {
    std::vector<MortonPrimitive<uint64_t>> primitives;
    
    for (uint32_t i = 0; i < 1000000; i++) {
        uint64_t code = /* your Morton code generation */;
        primitives.push_back({i, code});
    }
    
    tf::Executor executor{std::thread::hardware_concurrency()};
    
    ChunkedRadixSort(executor, primitives);
    return 0;
}
```

LBVH
```c++
#include "bvh/BVH.h"
#include <vector>

int main() {
    // Create a vector of primitives (e.g., triangles)
    std::vector<Primitive> primitives;
    
    Primitive triangle;
    float v1[3] = {0.0f, 0.0f, 0.0f};
    float v2[3] = {1.0f, 0.0f, 0.0f};
    float v3[3] = {0.0f, 1.0f, 0.0f};
    triangle.bounds.expand(v1);
    triangle.bounds.expand(v2);
    triangle.bounds.expand(v3);
    primitives.push_back(triangle);
    
    // Add more primitives...
    
    tf::Executor executor{std::thread::hardware_concurrency()};
    
    std::vector<BVHNode> bvh = buildLBVH<uint64_t>(executor, primitives);
    
    return 0;
}
```

Octree
```c++
#include "octree/Octree.h"
#include "util/Hilbert.h"
#include <vector>

int main() {
    std::vector<float> x = {1.0f, 2.0f, 3.0f, /* more points */ };
    std::vector<float> y = {0.5f, 1.5f, 2.5f, /* more points */ };
    std::vector<float> z = {0.0f, 1.0f, 2.0f, /* more points */ };
    
    Box<float> box;
    for (size_t i = 0; i < x.size(); i++) {
        Vec3<float> point(x[i], y[i], z[i]);
        box.expand(point);
    }
    
    using KeyType = uint64_t;
    size_t numPoints = x.size();
    std::vector<KeyType> codes(numPoints);
    
    tf::Executor executor{std::thread::hardware_concurrency()};
    
    computeSfcKeys(x.data(), y.data(), z.data(), codes.data(), numPoints, box, executor);
    
    // sort SFC keys(Optional)
    std::sort(codes.begin(), codes.end());
    
    unsigned bucketSize = 16; //example bucket size
    cstone::Octree<KeyType> octree(bucketSize);
    octree.build(codes.data(), codes.data() + codes.size(), executor);
    
    const auto& tree = octree.cornerstone();
    const auto& counts = octree.counts();
    const auto view = octree.view();
    
    return 0;
}
```

## References

- Sebastian Keller, Aurélien Cavelan, Rubén Cabezon, Lucio Mayer, Florina M. Ciorba.  
  *Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations*.  
  arXiv:2307.06345 [astro-ph.IM], 2023. https://arxiv.org/abs/2307.06345

- Theo Karras, [*Thinking Parallel, Part III: Tree Construction on the GPU*](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/), NVIDIA Blog.
