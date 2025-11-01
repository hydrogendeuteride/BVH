#ifndef BVH2_BOX2D_H
#define BVH2_BOX2D_H

#include <algorithm>
#include <limits>
#include <type_traits>

template<typename T>
struct Vec2
{
    T x, y;

    Vec2() : x(0), y(0) {}
    Vec2(T x_, T y_) : x(x_), y(y_) {}

    T &operator[](int i) { return i == 0 ? x : y; }
    const T &operator[](int i) const { return i == 0 ? x : y; }

    Vec2 operator+(const Vec2 &o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2 &o) const { return {x - o.x, y - o.y}; }
    Vec2 &operator+=(const Vec2 &o) { x += o.x; y += o.y; return *this; }
    Vec2 &operator-=(const Vec2 &o) { x -= o.x; y -= o.y; return *this; }
};

template<typename T>
struct Box2D
{
    Vec2<T> min;
    Vec2<T> max;

    Box2D()
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            min.x = min.y = std::numeric_limits<T>::max();
            max.x = max.y = std::numeric_limits<T>::lowest();
        }
        else
        {
            min.x = min.y = T(0);
            max.x = max.y = std::numeric_limits<T>::max();
        }
    }

    Box2D(const Vec2<T> &min_, const Vec2<T> &max_) : min(min_), max(max_) {}
    Box2D(T xmin, T xmax, T ymin, T ymax)
    {
        min.x = xmin; min.y = ymin; max.x = xmax; max.y = ymax;
    }

    void expand(const Vec2<T> &p)
    {
        min.x = std::min(min.x, p.x); max.x = std::max(max.x, p.x);
        min.y = std::min(min.y, p.y); max.y = std::max(max.y, p.y);
    }

    void centroid(T result[2]) const
    {
        result[0] = (min.x + max.x) / T(2);
        result[1] = (min.y + max.y) / T(2);
    }

    Vec2<T> centroid() const { return {(min.x + max.x) / T(2), (min.y + max.y) / T(2)}; }

    T xmin() const { return min.x; }
    T ymin() const { return min.y; }
    T xmax() const { return max.x; }
    T ymax() const { return max.y; }

    T width() const { return max.x - min.x; }
    T height() const { return max.y - min.y; }

    T ilx() const { return width() > 0 ? T(1) / width() : T(1); }
    T ily() const { return height() > 0 ? T(1) / height() : T(1); }

    static Box2D<T> unionOf(const Box2D<T> &a, const Box2D<T> &b)
    {
        Box2D<T> r;
        r.min.x = std::min(a.min.x, b.min.x);
        r.min.y = std::min(a.min.y, b.min.y);
        r.max.x = std::max(a.max.x, b.max.x);
        r.max.y = std::max(a.max.y, b.max.y);
        return r;
    }

    bool contains(const Box2D<T> &o) const
    {
        return min.x <= o.min.x && max.x >= o.max.x &&
               min.y <= o.min.y && max.y >= o.max.y;
    }
};

using Box2f = Box2D<float>;

template<typename KeyType>
struct IBox2D
{
    KeyType xmin, xmax;
    KeyType ymin, ymax;
    IBox2D(KeyType xmin_, KeyType xmax_, KeyType ymin_, KeyType ymax_)
            : xmin(xmin_), xmax(xmax_), ymin(ymin_), ymax(ymax_) {}
};

#endif // BVH2_BOX2D_H

