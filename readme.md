# BVH2

Fast C++17 header-only library for LBVH, octree, and quadtree construction with Taskflow
parallelism.

## Features

- LBVH build for `TriangleT<Scalar>` primitives (`bvh2::Primitive`, `bvh2::PrimitiveD`)
- Deterministic LBVH behavior for duplicated Morton codes (tie-break by primitive index)
- Selectable Morton sort backend: `bvh2::MortonSortMethod::StdSort` or `RadixSort`
- Traversal helpers:
  - `bvh2::traverseBVH`
  - `cstone::traverseOctree`
  - `qtree2d::traverseQuadtree`
- Parallel Morton primitive sort utility: `bvh2::ChunkedRadixSort`

## Requirements

- C++17
- CMake 3.15+
- Taskflow headers (`taskflow/taskflow.hpp`)

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### CMake options

- `BVH2_ENABLE_TESTS` (default: `ON`)
- `BVH2_ENABLE_OPENMP` (default: `ON`)
- `BVH2_ENABLE_AVX2` (default: `ON`)
- `BVH2_USE_LOCAL_TASKFLOW` (default: `ON`)
- `BVH2_TASKFLOW_INCLUDE_DIR` (optional Taskflow include directory)

Taskflow headers are resolved in this order:
1. `BVH2_TASKFLOW_INCLUDE_DIR`
2. Bundled `taskflow/` directory (when `BVH2_USE_LOCAL_TASKFLOW=ON`)
3. System include path (`find_path(taskflow/taskflow.hpp)`)

## Library integration

```cmake
add_subdirectory(BVH2)
target_link_libraries(your_target PRIVATE BVH2)
```

`BVH2` is an `INTERFACE` target, so there are no compiled library sources to link manually.

## Usage

### Parallel radix sort

```c++
#include "util/ParallelRadixSort.h"
#include <taskflow/taskflow.hpp>
#include <thread>
#include <vector>

int main()
{
    std::vector<bvh2::MortonPrimitive<uint64_t>> primitives;
    primitives.push_back({0, 20});
    primitives.push_back({1, 3});
    primitives.push_back({2, 12});

    tf::Executor executor(std::thread::hardware_concurrency());
    bvh2::ChunkedRadixSort(executor, primitives);
}
```

### LBVH build and traversal

```c++
#include "bvh/BVH.h"
#include <taskflow/taskflow.hpp>
#include <thread>
#include <vector>

int main()
{
    std::vector<bvh2::Primitive> primitives;
    primitives.emplace_back(Vec3<float>(0.0f, 0.0f, 0.0f),
                            Vec3<float>(1.0f, 0.0f, 0.0f),
                            Vec3<float>(0.0f, 1.0f, 0.0f));

    tf::Executor executor(std::thread::hardware_concurrency());
    auto nodes = bvh2::buildLBVH<uint64_t>(
            executor, primitives, bvh2::MortonSortMethod::RadixSort);

    bvh2::traverseBVH(nodes, [](uint32_t idx, const bvh2::BVHNode& node) {
        (void)idx;
        return !node.isLeaf; // return false to prune subtree
    });

    bvh2::Ray ray(Vec3<float>(-1.0f, 0.2f, 0.0f), Vec3<float>(1.0f, 0.0f, 0.0f));
    uint32_t hitIndex = 0;
    float hitT = 0.0f;
    bool hit = bvh2::traverseBVHClosestHit(nodes, primitives, ray, hitIndex, hitT);
    (void)hit;
}
```

### Double precision BVH

```c++
#include "bvh/BVH.h"
#include <taskflow/taskflow.hpp>
#include <thread>
#include <vector>

int main()
{
    std::vector<bvh2::PrimitiveD> primitives;
    primitives.emplace_back(Vec3<double>(0.0, 0.0, 0.0),
                            Vec3<double>(1.0, 0.0, 0.0),
                            Vec3<double>(0.0, 1.0, 0.0));

    tf::Executor executor(std::thread::hardware_concurrency());
    auto nodes = bvh2::buildLBVH<uint64_t>(executor, primitives);

    bvh2::RayD ray(Vec3<double>(-1.0, 0.2, 0.0), Vec3<double>(1.0, 0.0, 0.0));
    uint32_t hitIndex = 0;
    double hitT = 0.0;
    bool hit = bvh2::traverseBVHClosestHit(nodes, primitives, ray, hitIndex, hitT);
    (void)hit;
}
```

### Octree (3D)

```c++
#include "octree/Octree.h"
#include <algorithm>
#include <taskflow/taskflow.hpp>
#include <thread>
#include <vector>

int main()
{
    std::vector<float> x = {0.1f, 0.4f, 0.8f};
    std::vector<float> y = {0.3f, 0.5f, 0.9f};
    std::vector<float> z = {0.2f, 0.7f, 0.6f};

    Box<float> box;
    for (size_t i = 0; i < x.size(); ++i) box.expand(Vec3<float>(x[i], y[i], z[i]));

    std::vector<uint64_t> keys(x.size());
    tf::Executor executor(std::thread::hardware_concurrency());
    computeSfcKeys(x.data(), y.data(), z.data(), keys.data(), keys.size(), box, executor);
    std::sort(keys.begin(), keys.end());

    cstone::Octree<uint64_t> octree(16);
    octree.build(keys.data(), keys.data() + keys.size(), executor);

    auto view = octree.view();
    cstone::traverseOctree(view, [](cstone::TreeNodeIndex idx, uint64_t key, unsigned level) {
        (void)idx;
        (void)key;
        (void)level;
        return true;
    });
}
```

### Quadtree (2D)

```c++
#include "quadtree/Quadtree.h"
#include <algorithm>
#include <taskflow/taskflow.hpp>
#include <thread>
#include <vector>

int main()
{
    std::vector<float> x = {0.1f, 0.5f, 0.9f};
    std::vector<float> y = {0.2f, 0.4f, 0.8f};

    Box2D<float> box;
    for (size_t i = 0; i < x.size(); ++i) box.expand({x[i], y[i]});

    std::vector<uint64_t> keys(x.size());
    tf::Executor executor(std::thread::hardware_concurrency());
    computeSfcKeys2D<float, uint64_t>(x.data(), y.data(), keys.data(), keys.size(), box, executor);
    std::sort(keys.begin(), keys.end());

    qtree2d::Quadtree<uint64_t> quadtree(16);
    quadtree.build(keys.data(), keys.data() + keys.size(), executor);

    auto view = quadtree.view();
    qtree2d::traverseQuadtree(view, [](qtree2d::TreeNodeIndex idx, uint64_t key, unsigned level) {
        (void)idx;
        (void)key;
        (void)level;
        return true;
    });
}
```

## Tests and performance tests

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DBVH2_ENABLE_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Run individual binaries:

- `build/test/integration_test`
- `build/test/unit_test_radix_sort`
- `build/test/unit_test_BVH`
- `build/test/unit_test_octree`
- `build/test/unit_test_quadtree`
- `build/test/perf_test_radix_sort`
- `build/test/perf_test_BVH`
- `build/test/perf_test_octree`
- `build/test/perf_test_quadtree`

## References

- Sebastian Keller, Aurélien Cavelan, Rubén Cabezon, Lucio Mayer, Florina M. Ciorba.
  *Cornerstone: Octree Construction Algorithms for Scalable Particle Simulations*.
  arXiv:2307.06345 [astro-ph.IM], 2023. https://arxiv.org/abs/2307.06345
- Theo Karras, [*Thinking Parallel, Part III: Tree Construction on the GPU*](https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/), NVIDIA Blog.
