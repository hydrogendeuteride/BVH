#ifndef BVH2_BITOPS_H
#define BVH2_BITOPS_H

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace bvh2
{

template<typename KeyType>
struct unusedBits : std::integral_constant<unsigned,
                                           (std::numeric_limits<std::make_unsigned_t<KeyType>>::digits <= 32 ? 2u
                                                                                                               : 1u)>
{
    static_assert(std::is_unsigned_v<KeyType>, "KeyType must be unsigned");
};

template<class KeyType>
struct maxTreeLevel : std::integral_constant<unsigned,
                                             (std::numeric_limits<std::make_unsigned_t<KeyType>>::digits <= 32 ? 10u
                                                                                                                 : 21u)>
{
    static_assert(std::is_unsigned_v<KeyType>, "KeyType must be unsigned");
};

inline int countLeadingZeros(uint32_t x)
{
    constexpr int digits = std::numeric_limits<uint32_t>::digits;

    if (x == 0) return digits;

#if defined(_MSC_VER)
    unsigned long msbIndex = 0;
    _BitScanReverse(&msbIndex, static_cast<unsigned long>(x));
    return digits - 1 - static_cast<int>(msbIndex);
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_clz(x);
#else
    int count = 0;
    while ((x & (uint32_t(1) << (digits - 1))) == 0)
    {
        x <<= 1;
        ++count;
    }
    return count;
#endif
}

inline int countLeadingZeros(uint64_t x)
{
    constexpr int digits = std::numeric_limits<uint64_t>::digits;

    if (x == 0) return digits;

#if defined(_MSC_VER)
#if defined(_M_X64) || defined(_M_ARM64)
    unsigned long msbIndex = 0;
    _BitScanReverse64(&msbIndex, static_cast<unsigned __int64>(x));
    return digits - 1 - static_cast<int>(msbIndex);
#else
    unsigned long msbIndex = 0;
    uint32_t high = static_cast<uint32_t>(x >> 32);
    if (high)
    {
        _BitScanReverse(&msbIndex, static_cast<unsigned long>(high));
        return 31 - static_cast<int>(msbIndex);
    }

    _BitScanReverse(&msbIndex, static_cast<unsigned long>(x));
    return 63 - static_cast<int>(msbIndex);
#endif
#elif defined(__GNUC__) || defined(__clang__)
    return __builtin_clzll(x);
#else
    int count = 0;
    while ((x & (uint64_t(1) << (digits - 1))) == 0)
    {
        x <<= 1;
        ++count;
    }
    return count;
#endif
}

template<typename UInt>
inline std::enable_if_t<std::is_unsigned_v<UInt>, int> countLeadingZeros(UInt x)
{
    if constexpr (sizeof(UInt) <= sizeof(uint32_t))
    {
        return countLeadingZeros(static_cast<uint32_t>(x));
    }
    else
    {
        return countLeadingZeros(static_cast<uint64_t>(x));
    }
}

template<typename KeyType>
inline bool isPowerOf8(KeyType n)
{
    unsigned lz = countLeadingZeros(n - 1) - unusedBits<KeyType>{};
    return lz % 3 == 0 && !(n & (n - 1));
}

template<typename KeyType>
inline unsigned treeLevel(KeyType codeRange)
{
//    assert(isPowerOf8(codeRange));
    return (countLeadingZeros(codeRange - 1) - unusedBits<KeyType>{}) / 3;
}

template<typename KeyType>
inline KeyType nodeRange(unsigned treeLevel)
{
    assert(treeLevel <= maxTreeLevel<KeyType>{});
    unsigned shifts = maxTreeLevel<KeyType>{} - treeLevel;

    assert((3u * shifts) < std::numeric_limits<KeyType>::digits);
    return KeyType(1) << (3u * shifts);
}

template<typename KeyType>
inline unsigned octalDigit(KeyType code, unsigned position)
{
    return (code >> (3u * (maxTreeLevel<KeyType>{} - position))) & 7u;
}

template<typename KeyType>
inline KeyType encodePlaceholderBit(KeyType code, int prefixLength)
{
    int nShifts = 3 * maxTreeLevel<KeyType>{} - prefixLength;
    KeyType ret = code >> nShifts;
    KeyType placeHolderMask = KeyType(1) << prefixLength;

    return placeHolderMask | ret;
}

template<typename KeyType>
inline unsigned decodePrefixLength(KeyType code)
{
    return 8 * sizeof(KeyType) - 1 - countLeadingZeros(code);
}

template<typename KeyType>
inline KeyType decodePlaceholderBit(KeyType code)
{
    int prefixLength = decodePrefixLength(code);
    KeyType placeHolderMask = KeyType(1) << prefixLength;
    KeyType ret = code ^ placeHolderMask;

    return ret << (3 * maxTreeLevel<KeyType>{} - prefixLength);
}

template<typename KeyType>
inline int commonPrefix(KeyType key1, KeyType key2)
{
    return int(countLeadingZeros(key1 ^ key2)) - unusedBits<KeyType>{};
}

} // namespace bvh2

#endif //BVH2_BITOPS_H
