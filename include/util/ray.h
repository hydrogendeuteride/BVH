#ifndef RAY_H
#define RAY_H

#include "Box.h"
#include <limits>
#include <algorithm>
#include <cmath>

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

template<typename Scalar>
inline bool intersectRayAABB(const RayT<Scalar> &ray,
                             const Box<Scalar> &box,
                             Scalar tmin, Scalar tmax,
                             Scalar &tNear, Scalar &tFar)
{
    tNear = tmin;
    tFar = tmax;

    for (int axis = 0; axis < 3; ++axis)
    {
        Scalar invD = Scalar(1) / ray.direction[axis];
        Scalar t0 = (box.min[axis] - ray.origin[axis]) * invD;
        Scalar t1 = (box.max[axis] - ray.origin[axis]) * invD;

        if (invD < Scalar(0)) std::swap(t0, t1);

        tNear = std::max(tNear, t0);
        tFar = std::min(tFar, t1);

        if (tFar < tNear) return false;
    }

    return tFar >= tNear;
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
    const Scalar eps = static_cast<Scalar>(1e-8);

    Vec3<Scalar> edge1 = v1 - v0;
    Vec3<Scalar> edge2 = v2 - v0;

    Vec3<Scalar> pvec = cross(ray.direction, edge2);
    Scalar det = dot(edge1, pvec);

    if (std::abs(det) < eps)
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

#endif //RAY_H
