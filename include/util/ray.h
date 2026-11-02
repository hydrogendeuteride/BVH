#ifndef RAY_H
#define RAY_H

#include "BoundingBox.h"
#include <limits>
#include <algorithm>
#include <cmath>

namespace bvh2
{

template<typename Scalar>
struct RayT
{
    Vec3<Scalar> origin;
    Vec3<Scalar> direction;
    Scalar tmin;
    Scalar tmax;

    RayT()
            : origin(Scalar(0), Scalar(0), Scalar(0)),
              direction(Scalar(0), Scalar(0), Scalar(1)),
              tmin(Scalar(0)),
              tmax(std::numeric_limits<Scalar>::max())
    {}

    RayT(const Vec3<Scalar> &o, const Vec3<Scalar> &d,
         Scalar tmin_ = Scalar(0),
         Scalar tmax_ = std::numeric_limits<Scalar>::max())
            : origin(o), direction(d), tmin(tmin_), tmax(tmax_)
    {}
};

using Ray = RayT<float>;
using RayF = RayT<float>;
using RayD = RayT<double>;

template<typename Scalar>
inline Scalar dot(const Vec3<Scalar> &a, const Vec3<Scalar> &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<typename Scalar>
inline Vec3<Scalar> cross(const Vec3<Scalar> &a, const Vec3<Scalar> &b)
{
    return Vec3<Scalar>(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
    );
}

namespace detail
{
// Local to a traversal: RayT stays mutable and never carries a stale cache.
template<typename Scalar>
struct PreparedRayT
{
    RayT<Scalar> ray;
    Vec3<Scalar> inverseDirection;
    bool valid = true;
    bool regular = true;

    explicit PreparedRayT(const RayT<Scalar>& source) : ray(source)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!std::isfinite(ray.origin[axis]) || !std::isfinite(ray.direction[axis]))
            {
                valid = false;
                return;
            }
            inverseDirection[axis] = ray.direction[axis] == Scalar(0)
                    ? Scalar(0) : Scalar(1) / ray.direction[axis];
            regular = regular && ray.direction[axis] != Scalar(0) && std::isfinite(inverseDirection[axis]);
        }
    }
};

template<typename Scalar>
inline bool intersectPreparedRayAABB(const PreparedRayT<Scalar>& prepared,
                                     const Box<Scalar>& box,
                                     Scalar tmin, Scalar tmax,
                                     Scalar& tNear, Scalar& tFar)
{
    if (!prepared.valid || !(tmin <= tmax) || !std::isfinite(tmin) || !std::isfinite(tmax))
        return false;

    if (prepared.regular)
    {
        Vec3<Scalar> lower, upper;
        for (int axis = 0; axis < 3; ++axis)
        {
            if (!std::isfinite(box.min[axis]) || !std::isfinite(box.max[axis])) return false;
            const Scalar a = (box.min[axis] - prepared.ray.origin[axis]) * prepared.inverseDirection[axis];
            const Scalar b = (box.max[axis] - prepared.ray.origin[axis]) * prepared.inverseDirection[axis];
            lower[axis] = std::min(a, b);
            upper[axis] = std::max(a, b);
        }
        tNear = std::max(tmin, std::max(lower.x, std::max(lower.y, lower.z)));
        tFar = std::min(tmax, std::min(upper.x, std::min(upper.y, upper.z)));
        return tNear <= tFar;
    }

    tNear = tmin;
    tFar = tmax;
    for (int axis = 0; axis < 3; ++axis)
    {
        const Scalar origin = prepared.ray.origin[axis];
        const Scalar direction = prepared.ray.direction[axis];
        const Scalar boxMin = box.min[axis];
        const Scalar boxMax = box.max[axis];
        if (!std::isfinite(boxMin) || !std::isfinite(boxMax)) return false;
        // Keep exactly parallel rays out of the reciprocal arithmetic.
        if (direction == Scalar(0))
        {
            if (origin < boxMin || origin > boxMax) return false;
            continue;
        }
        const Scalar invD = prepared.inverseDirection[axis];
        // A nonzero subnormal direction can have an infinite reciprocal.
        const Scalar deltaMin = boxMin - origin, deltaMax = boxMax - origin;
        Scalar t0 = std::isfinite(invD) ? deltaMin * invD : deltaMin / direction;
        Scalar t1 = std::isfinite(invD) ? deltaMax * invD : deltaMax / direction;
        if (t0 > t1) std::swap(t0, t1);
        tNear = std::max(tNear, t0);
        tFar = std::min(tFar, t1);
        if (tNear > tFar) return false;
    }
    return tFar >= tNear;
}
// Two consecutive siblings share ray validation and per-axis ray loads.
template<typename Scalar>
inline unsigned intersectPreparedRayAABBPair(const PreparedRayT<Scalar>& prepared,
                                             const Box<Scalar>* boxes,
                                             Scalar tmin, Scalar tmax,
                                             Scalar& leftNear, Scalar& rightNear)
{
    if (!prepared.valid || !(tmin <= tmax) || !std::isfinite(tmin) || !std::isfinite(tmax)) return 0;
    if (!prepared.regular) {
        Scalar far;
        const bool left = intersectPreparedRayAABB(prepared, boxes[0], tmin, tmax, leftNear, far);
        const bool right = intersectPreparedRayAABB(prepared, boxes[1], tmin, tmax, rightNear, far);
        return unsigned(left) | (unsigned(right) << 1);
    }
    Scalar lower[3][2], upper[3][2];
    bool valid[2] = {true, true};
    for (int axis = 0; axis < 3; ++axis) {
        const Scalar origin = prepared.ray.origin[axis];
        const Scalar inverse = prepared.inverseDirection[axis];
        for (int child = 0; child < 2; ++child) {
            const Scalar minimum = boxes[child].min[axis], maximum = boxes[child].max[axis];
            valid[child] = valid[child] && std::isfinite(minimum) && std::isfinite(maximum);
            const Scalar a = (minimum - origin) * inverse;
            const Scalar b = (maximum - origin) * inverse;
            lower[axis][child] = std::min(a, b);
            upper[axis][child] = std::max(a, b);
        }
    }
    Scalar near[2];
    unsigned mask = 0;
    for (int child = 0; child < 2; ++child) {
        near[child] = std::max(tmin, std::max(lower[0][child], std::max(lower[1][child], lower[2][child])));
        const Scalar far = std::min(tmax, std::min(upper[0][child], std::min(upper[1][child], upper[2][child])));
        mask |= unsigned(valid[child] && near[child] <= far) << child;
    }
    leftNear = near[0]; rightNear = near[1];
    return mask;
}
} // namespace detail

template<typename Scalar>
inline bool intersectRayAABB(const RayT<Scalar>& ray,
                             const Box<Scalar>& box,
                             Scalar tmin, Scalar tmax,
                             Scalar& tNear, Scalar& tFar)
{
    return detail::intersectPreparedRayAABB(detail::PreparedRayT<Scalar>(ray), box,
                                           tmin, tmax, tNear, tFar);
}

// Convenience overloads for default float-based types
inline bool intersectRayAABB(const Ray &ray,
                             const BoundingBox &box,
                             float tmin, float tmax, float &tNear, float &tFar)
{
    return intersectRayAABB<float>(ray, box, tmin, tmax, tNear, tFar);
}

// Ray-triangle intersection using the Möller–Trumbore algorithm.
// Returns true if the ray hits the triangle within [tmin, tmax] and
// writes the hit distance into tHit.
template<typename Scalar>
inline bool intersectRayTriangle(const RayT<Scalar> &ray,
                                 const Vec3<Scalar> &v0,
                                 const Vec3<Scalar> &v1,
                                 const Vec3<Scalar> &v2,
                                 Scalar tmin, Scalar tmax,
                                 Scalar &tHit)
{
    Vec3<Scalar> edge1 = v1 - v0;
    Vec3<Scalar> edge2 = v2 - v0;

    Vec3<Scalar> pvec = cross(ray.direction, edge2);
    Scalar det = dot(edge1, pvec);

    // Scale the tolerance with the determinant's six product terms. A fixed
    // epsilon incorrectly rejects small triangles or short direction vectors.
    // Summing magnitudes before subtraction also accounts for cancellation.
    const Scalar detScale =
            std::abs(edge1.x) * (std::abs(ray.direction.y * edge2.z) + std::abs(ray.direction.z * edge2.y)) +
            std::abs(edge1.y) * (std::abs(ray.direction.z * edge2.x) + std::abs(ray.direction.x * edge2.z)) +
            std::abs(edge1.z) * (std::abs(ray.direction.x * edge2.y) + std::abs(ray.direction.y * edge2.x));
    const Scalar detTolerance = Scalar(8) * std::numeric_limits<Scalar>::epsilon() * detScale;

    if (!std::isfinite(det) || std::abs(det) <= detTolerance)
    {
        return false;
    }

    Scalar invDet = static_cast<Scalar>(1) / det;

    Vec3<Scalar> tvec = ray.origin - v0;
    Scalar u = dot(tvec, pvec) * invDet;
    if (u < static_cast<Scalar>(0) || u > static_cast<Scalar>(1))
    {
        return false;
    }

    Vec3<Scalar> qvec = cross(tvec, edge1);
    Scalar v = dot(ray.direction, qvec) * invDet;
    if (v < static_cast<Scalar>(0) || u + v > static_cast<Scalar>(1))
    {
        return false;
    }

    Scalar t = dot(edge2, qvec) * invDet;
    if (t < tmin || t > tmax)
    {
        return false;
    }

    tHit = t;
    return true;
}

// Triangle-specific convenience overload: pass three vertices directly.
template<typename Scalar>
inline bool intersectRayPrimitive(const RayT<Scalar> &ray,
                                  const Vec3<Scalar> &v0,
                                  const Vec3<Scalar> &v1,
                                  const Vec3<Scalar> &v2,
                                  Scalar tmin, Scalar tmax,
                                  Scalar &tHit)
{
    return intersectRayTriangle(ray, v0, v1, v2, tmin, tmax, tHit);
}

// Generic primitive intersection helper.
// By default, this treats the primitive as having an AABB member named `bounds`.
// You can customize this function (or specialize it) to intersect your actual
// primitive geometry instead of just its bounding box.
template<typename Scalar, typename Primitive>
inline bool intersectRayPrimitive(const RayT<Scalar> &ray,
                                  const Primitive &primitive,
                                  Scalar tmin, Scalar tmax,
                                  Scalar &tHit)
{
    Scalar tNear, tFar;
    if (!intersectRayAABB<Scalar>(ray, primitive.bounds, tmin, tmax, tNear, tFar))
    {
        return false;
    }

    tHit = tNear;
    return true;
}

} // namespace bvh2

#endif //RAY_H
