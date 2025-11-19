# BVH

Fast Linear Bounding Volume Hierarchy generation C++ header-only library with taskflow multithreading.

## Features

- LBVH (Linear Bounding Volume Hierarchy)
- Octree
- Quadtree

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
    // Create a vector of triangle primitives
    std::vector<Primitive> primitives;
    
    Primitive triangle;
    triangle.v0 = Vec3<float>(0.0f, 0.0f, 0.0f);
    triangle.v1 = Vec3<float>(1.0f, 0.0f, 0.0f);
    triangle.v2 = Vec3<float>(0.0f, 1.0f, 0.0f);
    triangle.updateBounds(); // compute AABB from triangle vertices
    primitives.push_back(triangle);
    
    // Add more primitives...
    
    tf::Executor executor{std::thread::hardware_concurrency()};

    // Choose sort method for Morton codes (StdSort or RadixSort)
    std::vector<BVHNode> bvh =
        buildLBVH<uint64_t>(executor, primitives, MortonSortMethod::RadixSort);

    // Simple closest-hit ray traversal
    Ray ray(Vec3<float>(-1.0f, 0.5f, 0.5f), Vec3<float>(1.0f, 0.0f, 0.0f));
    uint32_t hitIndex;
    float hitT;
    if (traverseBVHClosestHit(bvh, primitives, ray, hitIndex, hitT)) {
        // hitIndex is the index of the closest intersected primitive
    }
    
    return 0;
}
```

Double precision BVH
```c++
#include "bvh/BVH.h"
#include <vector>

int main() {
    std::vector<PrimitiveD> primitives;

    // One triangle in double precision
    PrimitiveD tri;
    tri.v0 = Vec3<double>(0.0, 0.0, 0.0);
    tri.v1 = Vec3<double>(1.0, 0.0, 0.0);
    tri.v2 = Vec3<double>(0.0, 1.0, 0.0);
    tri.updateBounds();
    primitives.push_back(tri);

    tf::Executor executor{std::thread::hardware_concurrency()};

    // BVH over double-precision triangles
    std::vector<BVHNodeD> bvh = buildLBVH<uint64_t>(executor, primitives);

    RayD ray(Vec3<double>(-1.0, 0.5, 0.5), Vec3<double>(1.0, 0.0, 0.0));
    uint32_t hitIndex;
    double hitT;
    if (traverseBVHClosestHit(bvh, primitives, ray, hitIndex, hitT)) {
        // hitIndex is the closest hit in double precision
    }

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
    
    std::sort(codes.begin(), codes.end());
    
    unsigned bucketSize = 16; //example bucket size
    cstone::Octree<KeyType> octree(bucketSize);
    octree.build(codes.data(), codes.data() + codes.size(), executor);
    
    const auto& tree = octree.cornerstone();
    const auto& counts = octree.counts();
    const auto view = octree.view();

    // Simple depth-first traversal over all nodes
    cstone::traverseOctree(view, [](cstone::TreeNodeIndex idx, KeyType key, unsigned level) {
        // idx: node index in view.prefixes, key: Morton/Hilbert key, level: tree depth
        (void)idx; (void)key; (void)level;
        return true; // return false to prune this subtree
    });
    
    return 0;
}
```

Quadtree (2D)
```c++
#include "quadtree/Quadtree.h"
#include "quadtree/Hilbert2D.h"
#include "util/Box2D.h"
#include <algorithm>

int main() {
    std::vector<float> x = {0.1f, 0.5f, 0.9f};
    std::vector<float> y = {0.2f, 0.4f, 0.8f};

    Box2D<float> box; for (size_t i=0;i<x.size();++i) box.expand({x[i],y[i]});

    using KeyType = uint64_t;
    std::vector<KeyType> keys(x.size());
    tf::Executor ex(1);
    computeSfcKeys2D<float,KeyType>(x.data(), y.data(), keys.data(), keys.size(), box, ex);
    std::sort(keys.begin(), keys.end());

    qtree2d::Quadtree<KeyType> qt(16);
    qt.build(keys.data(), keys.data()+keys.size(), ex);

    auto view = qt.view();

    // Simple depth-first traversal over all quadtree nodes
    qtree2d::traverseQuadtree(view, [](TreeNodeIndex idx, KeyType key, unsigned level) {
        (void)idx; (void)key; (void)level;
        return true;
    });

    return 0;
}
```

## Performance
System: AMD Ryzen 7 6800HS 8 core CPU, 32GB RAM, float32

|Structure| Input Number   | Build time (ms) | # Threads |
|--------|----------------|-----------------|-----------|
|Radix Sort| 1M uint64 keys | 30 ms           | 16        |
|BVH| 10K Triangles  | 1.87 ms         | 1         |
|BVH| 10K Triangles  | 1.64 ms         | 16        |
|BVH| 1M Triangles   | 157 ms          | 16        |
|Octree| 10K Points     | 3.36 ms         | 1         |
|Octree| 10K Points     | 6.09 ms         | 16        |
|Octree| 1M Points      | 102 ms          | 16        |
|Quadtree| 10K Points     | 4.81ms          | 1         |
|Quadtree| 1M Points      | 34.86ms         | 16        |


## References

- Sebastian Keller, Aurélien Cavelan, Rubén Cabezon, Lucio Mayer, Florina M. Ciorba.  
  *Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations*.  
  arXiv:2307.06345 [astro-ph.IM], 2023. https://arxiv.org/abs/2307.06345

- Theo Karras, [*Thinking Parallel, Part III: Tree Construction on the GPU*](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/), NVIDIA Blog.
