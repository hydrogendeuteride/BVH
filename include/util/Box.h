#ifndef BVH2_BOX_H
#define BVH2_BOX_H

#include <algorithm>
#include <limits>
#include <array>
#include <cmath>

template<typename T>
struct Vec3
{
    T x, y, z;

    Vec3() : x(0), y(0), z(0)
    {}

    Vec3(T x_, T y_, T z_) : x(x_), y(y_), z(z_)
    {}

    T &operator[](int i)
    {
        return i == 0 ? x : (i == 1 ? y : z);
    }

    const T &operator[](int i) const
    {
        return i == 0 ? x : (i == 1 ? y : z);
    }
};

template<typename T>
struct Box
{
    Vec3<T> min;
    Vec3<T> max;

    Box()
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            min.x = min.y = min.z = std::numeric_limits<T>::max();
            max.x = max.y = max.z = std::numeric_limits<T>::lowest();
        }
        else
        {
            min.x = min.y = min.z = 0;
            max.x = max.y = max.z = std::numeric_limits<T>::max();
        }
    }

    Box(const Vec3<T> &min_, const Vec3<T> &max_) : min(min_), max(max_)
    {}

    Box(T xmin, T xmax, T ymin, T ymax, T zmin, T zmax)
    {
        min.x = xmin;
        min.y = ymin;
        min.z = zmin;
        max.x = xmax;
        max.y = ymax;
        max.z = zmax;
    }

    void expand(const Vec3<T> &point)
    {
        min.x = std::min(min.x, point.x);
        min.y = std::min(min.y, point.y);
        min.z = std::min(min.z, point.z);

        max.x = std::max(max.x, point.x);
        max.y = std::max(max.y, point.y);
        max.z = std::max(max.z, point.z);
    }

    void expand(const T point[3])
    {
        min.x = std::min(min.x, point[0]);
        min.y = std::min(min.y, point[1]);
        min.z = std::min(min.z, point[2]);

        max.x = std::max(max.x, point[0]);
        max.y = std::max(max.y, point[1]);
        max.z = std::max(max.z, point[2]);
    }

    Vec3<T> centroid() const
    {
        return Vec3<T>(
                (min.x + max.x) / static_cast<T>(2),
                (min.y + max.y) / static_cast<T>(2),
                (min.z + max.z) / static_cast<T>(2)
        );
    }

    void centroid(T result[3]) const
    {
        result[0] = (min.x + max.x) / static_cast<T>(2);
        result[1] = (min.y + max.y) / static_cast<T>(2);
        result[2] = (min.z + max.z) / static_cast<T>(2);
    }

    T xmin() const
    { return min.x; }

    T ymin() const
    { return min.y; }

    T zmin() const
    { return min.z; }

    T xmax() const
    { return max.x; }

    T ymax() const
    { return max.y; }

    T zmax() const
    { return max.z; }

    T width() const
    { return max.x - min.x; }

    T height() const
    { return max.y - min.y; }

    T depth() const
    { return max.z - min.z; }

    //Inv length
    T ilx() const
    { return width() > 0 ? T(1) / width() : T(1); }

    T ily() const
    { return height() > 0 ? T(1) / height() : T(1); }

    T ilz() const
    { return depth() > 0 ? T(1) / depth() : T(1); }

    static Box<T> unionOf(const Box<T> &a, const Box<T> &b)
    {
        Box<T> result;
        result.min.x = std::min(a.min.x, b.min.x);
        result.min.y = std::min(a.min.y, b.min.y);
        result.min.z = std::min(a.min.z, b.min.z);

        result.max.x = std::max(a.max.x, b.max.x);
        result.max.y = std::max(a.max.y, b.max.y);
        result.max.z = std::max(a.max.z, b.max.z);

        return result;
    }

    bool contains(const Box<T> &other) const
    {
        return
                min.x <= other.min.x && max.x >= other.max.x &&
                min.y <= other.min.y && max.y >= other.max.y &&
                min.z <= other.min.z && max.z >= other.max.z;
    }
};

using BoxF = Box<float>;
using BoxI = Box<unsigned int>;

using BoundingBox = Box<float>;

struct IBox : public Box<unsigned int>
{
    using Box<unsigned int>::Box;

    IBox(unsigned xmin, unsigned xmax, unsigned ymin, unsigned ymax, unsigned zmin, unsigned zmax)
            : Box<unsigned int>(xmin, xmax, ymin, ymax, zmin, zmax)
    {}
};


#endif // BVH2_BOX_H
