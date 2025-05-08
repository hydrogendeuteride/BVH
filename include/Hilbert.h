#ifndef BVH2_HILBERT_H
#define BVH2_HILBERT_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <type_traits>
#include <utility>
#include <tuple>

namespace cstone
{
    struct IBox
    {
        unsigned xmin, xmax;
        unsigned ymin, ymax;
        unsigned zmin, zmax;

        constexpr IBox(unsigned xmi, unsigned xma, unsigned ymi, unsigned yma, unsigned zmi, unsigned zma)
                : xmin(xmi), xmax(xma), ymin(ymi), ymax(yma), zmin(zmi), zmax(zma)
        {
        }
    };

    template<class T>
    struct Vec3
    {
        T x, y, z;
    };

    template<class T>
    struct Box
    {
        Vec3<T> min;
        Vec3<T> max;
    };

    template<class KeyType>
    constexpr unsigned maxTreeLevel()
    {
        if constexpr (sizeof(KeyType) == 4)
            return 10;
        else
            return 20;
    }

    constexpr unsigned mortonToHilbert[8] = {0, 1, 3, 2, 7, 6, 4, 5};

    template<class KeyType>
    constexpr inline std::enable_if_t<std::is_unsigned_v<KeyType>, KeyType>
    iHilbert(unsigned px, unsigned py, unsigned pz) noexcept
    {
        assert(px < (1u << maxTreeLevel<KeyType>()));
        assert(py < (1u << maxTreeLevel<KeyType>()));
        assert(pz < (1u << maxTreeLevel<KeyType>()));

        KeyType key = 0;

        for (int level = maxTreeLevel<KeyType>() - 1; level >= 0; --level)
        {
            unsigned xi = (px >> level) & 1u;
            unsigned yi = (py >> level) & 1u;
            unsigned zi = (pz >> level) & 1u;

            unsigned octant = (xi << 2) | (yi << 1) | zi;
            key = (key << 3) + mortonToHilbert[octant];

            px ^= -(xi & ((!yi) | zi));
            py ^= -((xi & (yi | zi)) | (yi & (!zi)));
            pz ^= -((xi & (!yi) & (!zi)) | (yi & (!zi)));

            if (zi)
            {
                unsigned pt = px;
                px = py;
                py = pz;
                pz = pt;
            }
            else if (!yi)
            {
                unsigned pt = px;
                px = pz;
                pz = pt;
            }
        }

        return key;
    }

    template<class KeyType>
    inline std::tuple<unsigned, unsigned, unsigned> decodeHilbert(KeyType key) noexcept
    {
        unsigned px = 0;
        unsigned py = 0;
        unsigned pz = 0;

        for (unsigned level = 0; level < maxTreeLevel<KeyType>(); ++level)
        {
            unsigned octant = (key >> (3 * level)) & 7u;
            const unsigned xi = octant >> 2u;
            const unsigned yi = (octant >> 1u) & 1u;
            const unsigned zi = octant & 1u;

            if (yi ^ zi)
            {
                unsigned pt = px;
                px = pz;
                pz = py;
                py = pt;
            }
            else if ((!xi & !yi & !zi) || (xi & yi & zi))
            {
                unsigned pt = px;
                px = pz;
                pz = pt;
            }

            unsigned mask = (1 << level) - 1;
            px ^= mask & (-(xi & (yi | zi)));
            py ^= mask & (-((xi & ((!yi) | (!zi))) | ((!xi) & yi & zi)));
            pz ^= mask & (-((xi & (!yi) & (!zi)) | (yi & zi)));

            px |= (xi << level);
            py |= ((xi ^ yi) << level);
            pz |= ((yi ^ zi) << level);
        }

        return std::make_tuple(px, py, pz);
    }

    template<class KeyType>
    IBox hilbertIBox(KeyType keyStart, unsigned level) noexcept
    {
        assert(level <= maxTreeLevel<KeyType>());
        constexpr unsigned maxCoord = 1u << maxTreeLevel<KeyType>();
        unsigned cubeLength = maxCoord >> level;
        unsigned mask = ~(cubeLength - 1);

        auto coords = decodeHilbert<KeyType>(keyStart);
        unsigned ix = coords.first;
        unsigned iy = coords.second;
        unsigned iz = coords.third;

        ix &= mask;
        iy &= mask;
        iz &= mask;

        return IBox(ix, ix + cubeLength, iy, iy + cubeLength, iz, iz + cubeLength);
    }

    template<class KeyType, class T>
    inline KeyType hilbert3D(T x, T y, T z, T xmin, T ymin, T zmin, T mx, T my, T mz)
    {
        constexpr int mcoord = (1u << maxTreeLevel<KeyType>()) - 1;

        int ix = std::floor(x * mx) - xmin * mx;
        int iy = std::floor(y * my) - ymin * my;
        int iz = std::floor(z * mz) - zmin * mz;

        ix = std::min(ix, mcoord);
        iy = std::min(iy, mcoord);
        iz = std::min(iz, mcoord);

        assert(ix >= 0);
        assert(iy >= 0);
        assert(iz >= 0);

        return iHilbert<KeyType>(ix, iy, iz);
    }

    template<class KeyType, class T>
    inline KeyType hilbert3D(T x, T y, T z, const Box<T> &box)
    {
        constexpr unsigned cubeLength = (1u << maxTreeLevel<KeyType>());

        return hilbert3D<KeyType>(x, y, z, box.xmin(), box.ymin(), box.zmin(), cubeLength * box.ilx(),
                                  cubeLength * box.ily(), cubeLength * box.ilz());
    }

    template<class T, class KeyType>
    void computeSfcKeys(const T *x, const T *y, const T *z, KeyType *particleKeys, size_t n, const Box<T> &box)
    {
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i)
        {
            particleKeys[i] = hilbert3D<KeyType>(x[i], y[i], z[i], box);
        }
    }

    template<class KeyType, class T>
    std::tuple<Vec3<T>, Vec3<T>> centerAndSize(const Box<KeyType> &ibox, const Box<T> &box)
    {
        T normalization = T(1) / (KeyType(1) << (sizeof(KeyType) * 8 / 3));

        T xmin = box.min.x + (box.max.x - box.min.x) * ibox.xmin * normalization;
        T ymin = box.min.y + (box.max.y - box.min.y) * ibox.ymin * normalization;
        T zmin = box.min.z + (box.max.z - box.min.z) * ibox.zmin * normalization;

        T xmax = box.min.x + (box.max.x - box.min.x) * ibox.xmax * normalization;
        T ymax = box.min.y + (box.max.y - box.min.y) * ibox.ymax * normalization;
        T zmax = box.min.z + (box.max.z - box.min.z) * ibox.zmax * normalization;

        Vec3<T> center = {(xmin + xmax) * T(0.5), (ymin + ymax) * T(0.5), (zmin + zmax) * T(0.5)};
        Vec3<T> size = {(xmax - xmin) * T(0.5), (ymax - ymin) * T(0.5), (zmax - zmin) * T(0.5)};

        return {center, size};
    }
}

#endif // BVH2_HILBERT_H