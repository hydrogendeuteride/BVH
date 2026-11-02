#ifndef BVH2_REFLECTION_H
#define BVH2_REFLECTION_H

#include "BVH.h"
#include <cmath>

namespace bvh2
{

template<typename Scalar>
struct ReflectionHitT
{
    uint32_t primitiveIndex;
    uint32_t bounce; // Zero-based.
    Vec3<Scalar> position;
    Vec3<Scalar> normal; // Unit geometric normal, facing the incoming ray.
    Vec3<Scalar> incomingDirection;
    Vec3<Scalar> outgoingDirection;
    Scalar segmentLength;
    Scalar pathLength;
};

template<typename Scalar, typename Func>
uint32_t traceBVHReflections(const BVHView<Scalar> &nodes,
                             const std::vector<PrimitiveT<std::remove_const_t<Scalar>>> &primitives,
                             RayT<std::remove_const_t<Scalar>> ray,
                             uint32_t maxBounces,
                             Func &&visit,
                             std::remove_const_t<Scalar> relativeOriginOffset = std::remove_const_t<Scalar>(32) * std::numeric_limits<std::remove_const_t<Scalar>>::epsilon())
{
    using S = std::remove_const_t<Scalar>;
    Vec3<S> previousPosition = ray.origin;
    S pathLength = 0;
    uint32_t completed = 0;
    using VisitReturn = std::invoke_result_t<Func &, const ReflectionHitT<S> &>;

    for (uint32_t bounce = 0; bounce < maxBounces; ++bounce)
    {
        uint32_t primitiveIndex;
        S t;
        if (!traverseBVHClosestHit(nodes, primitives, ray, primitiveIndex, t)) break;

        const auto &triangle = primitives[primitiveIndex];
        const auto edge1 = triangle.v1 - triangle.v0;
        const auto edge2 = triangle.v2 - triangle.v0;
        auto normal = cross(edge1, edge2);
        const S normalLength = std::hypot(normal.x, normal.y, normal.z);
        if (!(normalLength > S(0)) || !std::isfinite(normalLength)) break;
        for (int axis = 0; axis < 3; ++axis) normal[axis] /= normalLength;
        if (dot(ray.direction, normal) > S(0))
            for (int axis = 0; axis < 3; ++axis) normal[axis] = -normal[axis];

        Vec3<S> position;
        Vec3<S> reflected;
        const S twiceProjection = S(2) * dot(ray.direction, normal);
        S positionScale = 0;
        for (int axis = 0; axis < 3; ++axis)
        {
            position[axis] = ray.origin[axis] + t * ray.direction[axis];
            reflected[axis] = ray.direction[axis] - twiceProjection * normal[axis];
            positionScale = std::max({positionScale, std::abs(position[axis]),
                                      std::abs(edge1[axis]), std::abs(edge2[axis])});
        }
        const auto segment = position - previousPosition;
        const S segmentLength = std::hypot(segment.x, segment.y, segment.z);
        pathLength += segmentLength;
        const ReflectionHitT<S> hit{primitiveIndex, bounce, position, normal,
                                         ray.direction, reflected, segmentLength, pathLength};
        ++completed;
        if constexpr (std::is_same_v<VisitReturn, bool>)
        {
            if (!visit(hit)) break;
        }
        else
        {
            visit(hit);
        }

        previousPosition = position;
        const S offset = std::max(S(0), relativeOriginOffset) * positionScale;
        for (int axis = 0; axis < 3; ++axis)
        {
            ray.origin[axis] = position[axis] + offset * normal[axis];
            // Ensure at least one representable step on each nonzero normal axis.
            if (normal[axis] != S(0))
                ray.origin[axis] = std::nextafter(ray.origin[axis],
                    std::copysign(std::numeric_limits<S>::infinity(), normal[axis]));
        }
        ray.direction = reflected;
        ray.tmin = 0;
    }
    return completed;
}

template<typename Scalar, typename Func>
uint32_t traceBVHReflections(const BVHData<Scalar>& nodes,
                             const std::vector<PrimitiveT<Scalar>>& primitives,
                             RayT<Scalar> ray, uint32_t maxBounces, Func&& visit,
                             Scalar relativeOriginOffset = Scalar(32) * std::numeric_limits<Scalar>::epsilon()) {
    return traceBVHReflections(nodes.data(), primitives, ray, maxBounces,
                               std::forward<Func>(visit), relativeOriginOffset);
}

} // namespace bvh2
#endif // BVH2_REFLECTION_H
