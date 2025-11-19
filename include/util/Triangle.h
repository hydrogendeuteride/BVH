#ifndef BVH2_TRIANGLE_H
#define BVH2_TRIANGLE_H

#include "Box.h"
#include "ray.h"

template<typename Scalar>
struct TriangleT
{
    Vec3<Scalar> v0;
    Vec3<Scalar> v1;
    Vec3<Scalar> v2;
    Box<Scalar> bounds;

    TriangleT()
            : v0(Scalar(0), Scalar(0), Scalar(0)),
              v1(Scalar(0), Scalar(0), Scalar(0)),
              v2(Scalar(0), Scalar(0), Scalar(0)),
              bounds()
    {
        bounds.expand(v0);
        bounds.expand(v1);
        bounds.expand(v2);
    }

    TriangleT(const Vec3<Scalar> &a,
              const Vec3<Scalar> &b,
              const Vec3<Scalar> &c)
            : v0(a), v1(b), v2(c), bounds()
    {
        bounds.expand(v0);
        bounds.expand(v1);
        bounds.expand(v2);
    }

    void updateBounds()
    {
        bounds = Box<Scalar>();
        bounds.expand(v0);
        bounds.expand(v1);
        bounds.expand(v2);
    }
};

// Triangle-specific primitive intersection: uses the Möller–Trumbore algorithm
// implemented in intersectRayTriangle (see ray.h).
template<typename Scalar>
inline bool intersectRayPrimitive(const RayT<Scalar> &ray,
                                  const TriangleT<Scalar> &tri,
                                  Scalar tmin, Scalar tmax,
                                  Scalar &tHit)
{
    return intersectRayTriangle<Scalar>(ray, tri.v0, tri.v1, tri.v2, tmin, tmax, tHit);
}

#endif // BVH2_TRIANGLE_H

