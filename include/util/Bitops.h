#ifndef BVH2_BITOPS_H
#define BVH2_BITOPS_H

#include <cassert>
#include <type_traits>
#include <cmath>
#include <cstdint>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

template<typename KeyType>
struct unusedBits
{
};

template<>
struct unusedBits<unsigned> : std::integral_constant<unsigned, 2>   //f32
{
};

template<>
struct unusedBits<unsigned long long> : std::integral_constant<unsigned, 1> //f64
{
};
template<>
struct unusedBits<unsigned long> : std::integral_constant<unsigned, 1> //f64
{
};

template<class KeyType>
struct maxTreeLevel
{
};

template<>
struct maxTreeLevel<unsigned> : std::integral_constant<unsigned, 10>
{
};

template<>
struct maxTreeLevel<unsigned long long> : std::integral_constant<unsigned, 21>
{
};
template<>
struct maxTreeLevel<unsigned long> : std::integral_constant<unsigned, 21>
{
};

constexpr int countLeadingZeros(uint32_t x)
{
#if defined(_MSC_VER)
    if (x == 0) return std::numeric_limits<uint32_t>::digits;
    return __lzcnt(x);
#elif defined(__GNUC__) || defined(__clang__)

    if (x == 0) return std::numeric_limits<uint32_t>::digits;
    return __builtin_clz(x);
#endif
}

constexpr int countLeadingZeros(uint64_t x)
{
#if defined(_MSC_VER)
    if (x == 0) return std::numeric_limits<uint64_t>::digits;
    return __lzcnt64(x);
#elif defined(__GNUC__) || defined(__clang__)
    if (x == 0) return std::numeric_limits<uint64_t>::digits;
    return __builtin_clzl(x);
#endif
}

template<typename KeyType>
constexpr bool isPowerOf8(KeyType n)
{
    unsigned lz = countLeadingZeros(n - 1) - unusedBits<KeyType>{};
    return lz % 3 == 0 && !(n & (n - 1));
}

template<typename KeyType>
constexpr unsigned treeLevel(KeyType codeRange)
{
//    assert(isPowerOf8(codeRange));
    return (countLeadingZeros(codeRange - 1) - unusedBits<KeyType>{}) / 3;
}

template<typename KeyType>
constexpr KeyType nodeRange(unsigned treeLevel)
{
    assert(treeLevel <= maxTreeLevel<KeyType>{});
    unsigned shifts = maxTreeLevel<KeyType>{} - treeLevel;

    return KeyType(1ul << (3u * shifts));
}

template<typename KeyType>
constexpr unsigned octalDigit(KeyType code, unsigned position)
{
    return (code >> (3u * (maxTreeLevel<KeyType>{} - position))) & 7u;
}

template<typename KeyType>
constexpr KeyType encodePlaceholderBit(KeyType code, int prefixLength)
{
    int nShifts = 3 * maxTreeLevel<KeyType>{} - prefixLength;
    KeyType ret = code >> nShifts;
    KeyType placeHolderMask = KeyType(1) << prefixLength;

    return placeHolderMask | ret;
}

template<typename KeyType>
constexpr unsigned decodePrefixLength(KeyType code)
{
    return 8 * sizeof(KeyType) - 1 - countLeadingZeros(code);
}

template<typename KeyType>
constexpr KeyType decodePlaceholderBit(KeyType code)
{
    int prefixLength = decodePrefixLength(code);
    KeyType placeHolderMask = KeyType(1) << prefixLength;
    KeyType ret = code ^ placeHolderMask;

    return ret << (3 * maxTreeLevel<KeyType>{} - prefixLength);
}

template<typename KeyType>
constexpr int commonPrefix(KeyType key1, KeyType key2)
{
    return int(countLeadingZeros(key1 ^ key2)) - unusedBits<KeyType>{};
}

#endif //BVH2_BITOPS_H
