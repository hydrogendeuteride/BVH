#ifndef RAY_H
#define RAY_H

#include "Box.h"
#include <limits>
#include <algorithm>

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

#endif //RAY_H
