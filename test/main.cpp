#include <iostream>
#include <iomanip>
#include "bvh/BVH.h"

void exampleUsage()
{
    std::vector<Primitive> primitives;

    Primitive triangle1;
    float v1[3] = {0.0f, 0.0f, 0.0f};
    float v2[3] = {1.0f, 0.0f, 0.0f};
    float v3[3] = {0.0f, 1.0f, 0.0f};
    triangle1.bounds.expand(v1);
    triangle1.bounds.expand(v2);
    triangle1.bounds.expand(v3);
    primitives.push_back(triangle1);

    Primitive triangle2;
    float v4[3] = {0.0f, 0.0f, 1.0f};
    float v5[3] = {1.0f, 0.0f, 1.0f};
    float v6[3] = {0.0f, 1.0f, 1.0f};
    triangle2.bounds.expand(v4);
    triangle2.bounds.expand(v5);
    triangle2.bounds.expand(v6);
    primitives.push_back(triangle2);

    Primitive triangle3;
    float v7[3] = {2.0f, 2.0f, 0.0f};
    float v8[3] = {3.0f, 2.0f, 0.0f};
    float v9[3] = {2.0f, 3.0f, 1.0f};
    triangle3.bounds.expand(v7);
    triangle3.bounds.expand(v8);
    triangle3.bounds.expand(v9);
    primitives.push_back(triangle3);

    for (int i = 0; i < 10; ++i)
    {
        Primitive triangle;
        float offset = static_cast<float>(i) * 0.5f;
        float va[3] = {offset, offset, offset};
        float vb[3] = {offset + 1.0f, offset, offset};
        float vc[3] = {offset, offset + 1.0f, offset};
        triangle.bounds.expand(va);
        triangle.bounds.expand(vb);
        triangle.bounds.expand(vc);
        primitives.push_back(triangle);
    }

    std::cout << "Building LBVH for " << primitives.size() << " primitives..." << std::endl;

    std::vector<BVHNode> bvh = buildLBVH(primitives);

    std::cout << "LBVH built successfully with " << bvh.size() << " nodes" << std::endl;

    std::cout << "\n=== BVH Node Details ===\n";

    uint32_t numPrimitives = primitives.size();
    uint32_t numInternalNodes = numPrimitives - 1;

    for (size_t i = 0; i < bvh.size(); ++i)
    {
        const BVHNode &node = bvh[i];
        bool isInternal = (node.object_idx == 0xFFFFFFFF);

        std::cout << "\nNode " << i << (isInternal ? " (Internal)" : " (Leaf)") << ":" << std::endl;

        std::cout << "  Bounds: min("
                  << node.bounds.min[0] << ", " << node.bounds.min[1] << ", " << node.bounds.min[2]
                  << "), max("
                  << node.bounds.max[0] << ", " << node.bounds.max[1] << ", " << node.bounds.max[2]
                  << ")" << std::endl;

        if (isInternal)
        {
            std::cout << "  Left child: " << node.left_idx;
            std::cout << " (" << (bvh[node.left_idx].object_idx == 0xFFFFFFFF ? "Internal" : "Leaf") << ")"
                      << std::endl;

            std::cout << "  Right child: " << node.right_idx;
            std::cout << " (" << (bvh[node.right_idx].object_idx == 0xFFFFFFFF ? "Internal" : "Leaf") << ")"
                      << std::endl;
        }
        else
        {
            std::cout << "  Object index: " << node.object_idx << std::endl;
        }

        if (i > 0 || node.parent_idx != 0)
        {
            std::cout << "  Parent: " << node.parent_idx << std::endl;
        }
        else
        {
            std::cout << "  Parent: None (Root)" << std::endl;
        }

        float volume = (node.bounds.max[0] - node.bounds.min[0]) *
                       (node.bounds.max[1] - node.bounds.min[1]) *
                       (node.bounds.max[2] - node.bounds.min[2]);

        float surface_area = 2.0f * (
                (node.bounds.max[0] - node.bounds.min[0]) * (node.bounds.max[1] - node.bounds.min[1]) +
                (node.bounds.max[0] - node.bounds.min[0]) * (node.bounds.max[2] - node.bounds.min[2]) +
                (node.bounds.max[1] - node.bounds.min[1]) * (node.bounds.max[2] - node.bounds.min[2])
        );

        std::cout << "  Volume: " << volume << std::endl;
        std::cout << "  Surface Area: " << surface_area << std::endl;
    }

    std::cout << "\n=== BVH Tree Structure ===\n";

    std::function<void(uint32_t, int)> printTreeRecursive = [&](uint32_t nodeIdx, int depth) {
        const BVHNode &node = bvh[nodeIdx];

        for (int i = 0; i < depth; i++)
        {
            std::cout << "  ";
        }

        if (node.object_idx == 0xFFFFFFFF)
        {
            std::cout << "Node " << nodeIdx << " (Internal)" << std::endl;
            printTreeRecursive(node.left_idx, depth + 1);
            printTreeRecursive(node.right_idx, depth + 1);
        }
        else
        {
            std::cout << "Node " << nodeIdx << " (Leaf, Object " << node.object_idx << ")" << std::endl;
        }
    };

    printTreeRecursive(0, 0);

    std::cout << "\n=== BVH Tree Statistics ===\n";

    int maxDepth = 0;
    int leafCount = 0;
    int sumDepth = 0;
    std::vector<int> nodesPerLevel;

    std::function<void(uint32_t, int)> analyzeNode = [&](uint32_t nodeIdx, int depth) {
        if (depth > maxDepth)
        {
            maxDepth = depth;
        }

        if (static_cast<size_t>(depth) >= nodesPerLevel.size())
        {
            nodesPerLevel.resize(depth + 1, 0);
        }

        nodesPerLevel[depth]++;

        const BVHNode &node = bvh[nodeIdx];

        if (node.object_idx != 0xFFFFFFFF)
        {
            leafCount++;
            sumDepth += depth;
        }
        else
        {
            analyzeNode(node.left_idx, depth + 1);
            analyzeNode(node.right_idx, depth + 1);
        }
    };

    analyzeNode(0, 0);

    float avgLeafDepth = static_cast<float>(sumDepth) / leafCount;

    std::cout << "Total nodes: " << bvh.size() << std::endl;
    std::cout << "Internal nodes: " << bvh.size() - leafCount << std::endl;
    std::cout << "Leaf nodes: " << leafCount << std::endl;
    std::cout << "Maximum tree depth: " << maxDepth << std::endl;
    std::cout << "Average leaf depth: " << avgLeafDepth << std::endl;

    std::cout << "\nNodes per level:" << std::endl;
    for (size_t level = 0; level < nodesPerLevel.size(); level++)
    {
        std::cout << "  Level " << level << ": " << nodesPerLevel[level] << " nodes" << std::endl;
    }

    std::cout << "\n=== Morton Code Analysis ===\n";

    std::vector<MortonPrimitive> mortonPrimitives = generateMortonCodes(primitives);

    const size_t numCodesToShow = std::min(size_t(10), mortonPrimitives.size());
    std::cout << "First " << numCodesToShow << " Morton codes:" << std::endl;

    for (size_t i = 0; i < numCodesToShow; i++)
    {
        std::cout << "  Primitive " << mortonPrimitives[i].primitiveIndex
                  << ": 0x" << std::hex << mortonPrimitives[i].mortonCode << std::dec << std::endl;
    }

    if (!mortonPrimitives.empty())
    {
        std::vector<int> setBitCounts(64, 0);

        for (const auto &mp: mortonPrimitives)
        {
            uint64_t code = mp.mortonCode;
            for (int bit = 0; bit < 64; bit++)
            {
                if (code & (1ULL << bit))
                {
                    setBitCounts[bit]++;
                }
            }
        }

        std::cout << "\nMorton code bit distribution (percentage of codes with bit set):" << std::endl;
        for (int bit = 63; bit >= 0; bit--)
        {
            if (bit % 8 == 7)
            {
                std::cout << "\n  Bits " << bit - 7 << "-" << bit << ": ";
            }
            float percentage = 100.0f * setBitCounts[bit] / mortonPrimitives.size();
            std::cout << std::fixed << std::setprecision(1) << percentage << "% ";
        }
        std::cout << std::endl;
    }
}

int main()
{
    exampleUsage();
    return 0;
}