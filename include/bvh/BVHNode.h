#ifndef BVH2_BVHNODE_H
#define BVH2_BVHNODE_H

#include "util/BoundingBox.h"
#include "util/Triangle.h"
#include <cstdint>
#include <cstddef>
#include <type_traits>
#include <vector>

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

template<typename Scalar>
struct BVHView
{
    using Value = std::remove_const_t<Scalar>;
    using Bounds = std::conditional_t<std::is_const_v<Scalar>, const Box<Value>, Box<Value>>;
    using Index = std::conditional_t<std::is_const_v<Scalar>, const uint32_t, uint32_t>;
    uint32_t numNodes = 0;
    Bounds* bounds = nullptr;
    Index* childOffsets = nullptr;
    Index* primitiveIndices = nullptr;

    bool empty() const { return numNodes == 0; }
    size_t size() const { return numNodes; }
    bool isLeaf(uint32_t i) const { return primitiveIndices[i] != UINT32_MAX; }
    BVHNodeT<Value> operator[](uint32_t i) const {
        return {bounds[i], childOffsets[i], primitiveIndices[i]};
    }
};

template<typename Scalar>
class BVHData
{
public:
    void resize(uint32_t n) {
        bounds.resize(n);
        childOffsets.resize(n, UINT32_MAX);
        primitiveIndices.resize(n, UINT32_MAX);
    }
    BVHView<Scalar> data() {
        return {uint32_t(size()), bounds.data(), childOffsets.data(), primitiveIndices.data()};
    }
    BVHView<const Scalar> data() const {
        return {uint32_t(size()), bounds.data(), childOffsets.data(), primitiveIndices.data()};
    }
    bool empty() const { return bounds.empty(); }
    size_t size() const { return bounds.size(); }
    BVHNodeT<Scalar> operator[](uint32_t i) const { return data()[i]; }

    std::vector<Box<Scalar>> bounds;
    std::vector<uint32_t> childOffsets;
    std::vector<uint32_t> primitiveIndices;
};

} // namespace bvh2

#endif // BVH2_BVHNODE_H
