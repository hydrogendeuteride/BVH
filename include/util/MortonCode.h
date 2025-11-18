#ifndef BVH2_MORTONCODE_H
#define BVH2_MORTONCODE_H

#include <cstdint>
#include <algorithm>

template<typename MortonCodeType>
struct MortonPrimitive
{
    uint32_t primitiveIndex;
    MortonCodeType mortonCode;

    bool operator==(const MortonPrimitive &other) const
    {
        return primitiveIndex == other.primitiveIndex && mortonCode == other.mortonCode;
    }
};

template<typename T>
inline T expandBits(uint32_t v)
{
    T result = static_cast<T>(v);

    if constexpr (sizeof(T) <= 4)
    {
        result = (result * 0x00010001u) & 0xFF0000FFu;
        result = (result * 0x00000101u) & 0x0F00F00Fu;
        result = (result * 0x00000011u) & 0xC30C30C3u;
        result = (result * 0x00000005u) & 0x49249249u;
    }
    else
    {
        result = (result * 0x0001000100010001ULL) & 0xFF0000FF0000FFULL;
        result = (result * 0x0000010100000101ULL) & 0x0F00F00F00F00F0FULL;
        result = (result * 0x0000000100000001ULL) & 0xC30C30C30C30C30CULL;
        result = (result * 0x0000000500000005ULL) & 0x4924924949249249ULL;
    }

    return result;
}

template<typename MortonCodeType, typename Real>
inline MortonCodeType computeMortonCode(const Real pos[3], const Real sceneMin[3], const Real sceneExtent[3])
{
    Real normalized[3];
    for (int i = 0; i < 3; ++i)
    {
        Real v = (pos[i] - sceneMin[i]) / sceneExtent[i];
        if (v < Real(0)) v = Real(0);
        if (v > Real(1)) v = Real(1);
        normalized[i] = v;
    }

    constexpr uint32_t maxCoordValue = (sizeof(MortonCodeType) <= 4) ? 1023 : 2097151;

    uint32_t x = static_cast<uint32_t>(normalized[0] * maxCoordValue);
    uint32_t y = static_cast<uint32_t>(normalized[1] * maxCoordValue);
    uint32_t z = static_cast<uint32_t>(normalized[2] * maxCoordValue);

    return (expandBits<MortonCodeType>(z) << 2) | (expandBits<MortonCodeType>(y) << 1) | expandBits<MortonCodeType>(x);
}

inline uint64_t expandBits(uint32_t v)
{
    return expandBits<uint64_t>(v);
}

#endif //BVH2_MORTONCODE_H
