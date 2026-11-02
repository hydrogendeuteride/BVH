#ifndef BVH2_BVHNODE_H
#define BVH2_BVHNODE_H

#include "util/BoundingBox.h"
#include "util/Triangle.h"
#include <cstdint>

namespace bvh2
{

template<typename Scalar>
using PrimitiveT = TriangleT<Scalar>;

using Primitive = PrimitiveT<float>;
using PrimitiveF = PrimitiveT<float>;
using PrimitiveD = PrimitiveT<double>;

template<typename Scalar>
struct BVHNodeT
{
    Box<Scalar> bounds;
    uint32_t left_idx = UINT32_MAX;
    uint32_t object_idx = UINT32_MAX;

    bool isLeaf() const { return object_idx != UINT32_MAX; }
    uint32_t right_idx() const { return left_idx + 1; }
};

using BVHNode = BVHNodeT<float>;
using BVHNodeF = BVHNodeT<float>;
using BVHNodeD = BVHNodeT<double>;

static_assert(sizeof(BVHNodeF) == 32);
static_assert(sizeof(BVHNodeD) == 56);

} // namespace bvh2

#endif // BVH2_BVHNODE_H
