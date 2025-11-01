#ifndef BVH2_HILBERT2D_H
#define BVH2_HILBERT2D_H

#include <tuple>
#include <type_traits>
#include <cassert>
#include <cmath>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>

#include "util/Box2D.h"
#include "quadtree/Bitops2D.h"

// Integer Hilbert encode (2D). px,py in [0, 2^L - 1], L = maxTreeLevel2D
template<typename KeyType>
inline std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
iHilbert2D(unsigned px, unsigned py) noexcept
{
    const unsigned L = maxTreeLevel2D<KeyType>{};
    const unsigned n = 1u << L;
    assert(px < n);
    assert(py < n);

    unsigned x = px;
    unsigned y = py;
    KeyType key = 0;

    for (unsigned s = n >> 1; s > 0; s >>= 1)
    {
        unsigned rx = (x & s) ? 1u : 0u;
        unsigned ry = (y & s) ? 1u : 0u;
        unsigned digit = (3u * rx) ^ ry;     // quadrant id in Hilbert order at this level
        key = (key << 2) | digit;

        if (ry == 0)
        {
            if (rx == 1u)
            {
                x = n - 1 - x;
                y = n - 1 - y;
            }
            unsigned t = x; x = y; y = t;
        }
    }

    return key;
}

// Decode integer Hilbert key (2D) into grid coordinates.
template<typename KeyType>
inline std::tuple<unsigned, unsigned> decodeHilbert2D(KeyType key) noexcept
{
    const unsigned L = maxTreeLevel2D<KeyType>{};
    const unsigned n = 1u << L;

    unsigned x = 0;
    unsigned y = 0;
    KeyType t = key;

    for (unsigned s = 1; s < n; s <<= 1)
    {
        unsigned rx = (t >> 1) & 1u;
        unsigned ry = (t ^ rx) & 1u;

        if (ry == 0)
        {
            if (rx == 1u)
            {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            unsigned tmp = x; x = y; y = tmp;
        }

        x += s * rx;
        y += s * ry;
        t >>= 2;
    }

    return std::make_tuple(x, y);
}

template<typename KeyType>
inline IBox2D<KeyType> hilbertIBox2D(KeyType keyStart, unsigned level) noexcept
{
    const unsigned maxCoord = 1u << maxTreeLevel2D<KeyType>();
    unsigned side = maxCoord >> level;
    unsigned mask = ~(side - 1u);

    auto coords = decodeHilbert2D<KeyType>(keyStart);
    unsigned ix = std::get<0>(coords) & mask;
    unsigned iy = std::get<1>(coords) & mask;

    return {KeyType(ix), KeyType(ix + side), KeyType(iy), KeyType(iy + side)};
}

template<typename KeyType, typename T>
inline KeyType hilbert2D(T x, T y, T xmin, T ymin, T mx, T my)
{
    constexpr int mcoord = (1u << maxTreeLevel2D<KeyType>()) - 1;

    int ix = int(std::floor(x * mx)) - int(std::floor(xmin * mx));
    int iy = int(std::floor(y * my)) - int(std::floor(ymin * my));

    ix = std::min(ix, mcoord); iy = std::min(iy, mcoord);
    ix = std::max(ix, 0);      iy = std::max(iy, 0);

    return iHilbert2D<KeyType>(unsigned(ix), unsigned(iy));
}

template<typename KeyType, typename T>
inline KeyType hilbert2D(T x, T y, const Box2D<T> &box)
{
    const unsigned grid = 1u << maxTreeLevel2D<KeyType>();
    return hilbert2D<KeyType>(x, y, box.xmin(), box.ymin(), grid * box.ilx(), grid * box.ily());
}

template<typename T, typename KeyType>
inline std::tuple<Vec2<T>, Vec2<T>> centerAndSize2D(const IBox2D<KeyType> &ibox, const Box2D<T> &box)
{
    const T norm = T(1) / T(1u << maxTreeLevel2D<KeyType>());

    T xmin = box.min.x + (box.max.x - box.min.x) * ibox.xmin * norm;
    T ymin = box.min.y + (box.max.y - box.min.y) * ibox.ymin * norm;
    T xmax = box.min.x + (box.max.x - box.min.x) * ibox.xmax * norm;
    T ymax = box.min.y + (box.max.y - box.min.y) * ibox.ymax * norm;

    Vec2<T> center{(xmin + xmax) * T(0.5), (ymin + ymax) * T(0.5)};
    Vec2<T> size  {(xmax - xmin) * T(0.5), (ymax - ymin) * T(0.5)};

    return {center, size};
}

template<typename T, typename KeyType>
inline void computeSfcKeys2D(const T *x, const T *y, KeyType *keys,
                             size_t n, const Box2D<T> &box, tf::Executor &executor)
{
    tf::Taskflow tf;
    tf.for_each_index(size_t(0), n, size_t(1), [&](size_t i) {
        keys[i] = hilbert2D<KeyType>(x[i], y[i], box);
    });
    executor.run(tf).wait();
}

#endif // BVH2_HILBERT2D_H
