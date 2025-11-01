#ifndef BVH2_BITOPS2D_H
#define BVH2_BITOPS2D_H

#include <type_traits>
#include <cassert>
#include <cstdint>

#include "util/Bitops.h" // for countLeadingZeros

// Unused top bits to make space for the placeholder bit scheme
template<typename KeyType>
struct unusedBits2D {};

template<>
struct unusedBits2D<unsigned> : std::integral_constant<unsigned, 2> {};

template<>
struct unusedBits2D<unsigned long long> : std::integral_constant<unsigned, 2> {};

template<>
struct unusedBits2D<unsigned long> : std::integral_constant<unsigned, 2> {};

// Maximum quadtree levels (2 bits per level)
template<typename KeyType>
struct maxTreeLevel2D {};

template<>
struct maxTreeLevel2D<unsigned> : std::integral_constant<unsigned, 15> {};

template<>
struct maxTreeLevel2D<unsigned long long> : std::integral_constant<unsigned, 31> {};

template<>
struct maxTreeLevel2D<unsigned long> : std::integral_constant<unsigned, 31> {};

template<typename KeyType>
constexpr bool isPowerOf4(KeyType n)
{
    unsigned lz = countLeadingZeros(n - 1) - unusedBits2D<KeyType>{};
    return (lz % 2 == 0) && !(n & (n - 1));
}

template<typename KeyType>
constexpr unsigned qtTreeLevel(KeyType codeRange)
{
//    assert(isPowerOf4(codeRange));
    return (countLeadingZeros(static_cast<uint64_t>(codeRange - 1)) - unusedBits2D<KeyType>{}) / 2;
}

template<typename KeyType>
constexpr KeyType nodeRange2D(unsigned level)
{
    assert(level <= maxTreeLevel2D<KeyType>{});
    unsigned shifts = maxTreeLevel2D<KeyType>{} - level;
    return KeyType(1ull << (2u * shifts));
}

template<typename KeyType>
constexpr unsigned quadDigit(KeyType code, unsigned position)
{
    return (code >> (2u * (maxTreeLevel2D<KeyType>{} - position))) & 3u;
}

template<typename KeyType>
constexpr KeyType encodePlaceholderBit2D(KeyType code, int prefixLength)
{
    int nShifts = 2 * maxTreeLevel2D<KeyType>{} - prefixLength;
    KeyType ret = code >> nShifts;
    KeyType placeHolderMask = KeyType(1) << prefixLength;
    return placeHolderMask | ret;
}

template<typename KeyType>
constexpr unsigned decodePrefixLength2D(KeyType code)
{
    return 8 * sizeof(KeyType) - 1 - countLeadingZeros(static_cast<uint64_t>(code));
}

template<typename KeyType>
constexpr KeyType decodePlaceholderBit2D(KeyType code)
{
    int prefixLength = decodePrefixLength2D<KeyType>(code);
    KeyType placeHolderMask = KeyType(1) << prefixLength;
    KeyType ret = code ^ placeHolderMask;
    return ret << (2 * maxTreeLevel2D<KeyType>{} - prefixLength);
}

template<typename KeyType>
constexpr int commonPrefix2D(KeyType key1, KeyType key2)
{
    return int(countLeadingZeros(static_cast<uint64_t>(key1 ^ key2))) - unusedBits2D<KeyType>{};
}

#endif // BVH2_BITOPS2D_H
