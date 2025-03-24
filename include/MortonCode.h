#ifndef BVH2_MORTONCODE_H
#define BVH2_MORTONCODE_H

#include <cstdint>
#include <algorithm>

struct MortonPrimitive
{
    uint32_t primitiveIndex;
    uint64_t mortonCode;

    bool operator==(const MortonPrimitive &other) const
    {
        return primitiveIndex == other.primitiveIndex && mortonCode == other.mortonCode;
    }
};

inline uint64_t expandBits(uint32_t v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

inline uint64_t computeMortonCode(const float pos[3], const float sceneMin[3], const float sceneExtent[3])
{
    float normalized[3];
    for (int i = 0; i < 3; ++i)
    {
        normalized[i] = std::min(std::max((pos[i] - sceneMin[i]) / sceneExtent[i], 0.0f), 1.0f);
    }

    uint32_t x = static_cast<uint32_t>(normalized[0] * 1023.0f);
    uint32_t y = static_cast<uint32_t>(normalized[1] * 1023.0f);
    uint32_t z = static_cast<uint32_t>(normalized[2] * 1023.0f);

    return (expandBits(z) << 2) | (expandBits(y) << 1) | expandBits(x);
}

#endif //BVH2_MORTONCODE_H
