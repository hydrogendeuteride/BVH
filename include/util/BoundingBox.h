#ifndef BVH2_BOUNDINGBOX_H
#define BVH2_BOUNDINGBOX_H

#include <algorithm>
#include <limits>

struct BoundingBox
{

    float min[3] = {std::numeric_limits<float>::max(), std::numeric_limits<float>::max(),
                    std::numeric_limits<float>::max()};
    float max[3] = {std::numeric_limits<float>::lowest(), std::numeric_limits<float>::lowest(),
                    std::numeric_limits<float>::lowest()};

    void expand(const float point[3])
    {
        for (int i = 0; i < 3; ++i)
        {
            min[i] = std::min(min[i], point[i]);
            max[i] = std::max(max[i], point[i]);
        }
    }

    void centroid(float result[3]) const
    {
        for (int i = 0; i < 3; ++i)
        {
            result[i] = (min[i] + max[i]) * 0.5f;
        }
    }
};

#endif //BVH2_BOUNDINGBOX_H
