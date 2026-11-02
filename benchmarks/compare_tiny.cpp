#define TINYBVH_IMPLEMENTATION
#include <tiny_bvh.h>
#include "bvh/BVH.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <thread>

using V = bvh2::Vec3<float>;
using Clock = std::chrono::steady_clock;
template<class F> double timed(F&& f) {
    const auto start = Clock::now();
    f();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}
V unit(V v) {
    const float length = std::sqrt(bvh2::dot(v, v));
    return {v.x / length, v.y / length, v.z / length};
}
struct Result { V end; float path = 0; unsigned hits = 0; };

int main(int argc, char** argv) {
    const int grid = argc > 1 ? std::atoi(argv[1]) : 32;
    const int rayCount = argc > 2 ? std::atoi(argv[2]) : 32768;
    const unsigned hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
    const unsigned workers = argc > 3 ? unsigned(std::atoi(argv[3])) : 1;
    if (grid < 1 || grid > 256 || rayCount < 1 || rayCount > 10000000) return 2;
    // TinyBVH's built-in pool uses hardware_concurrency - 1 workers plus the caller.
    if (workers != 1 && workers != hardwareThreads) {
        std::fprintf(stderr, "Thread count must be 1 or %u (TinyBVH pool size).\n", hardwareThreads);
        return 2;
    }
    tf::Executor executor(workers);
    std::mt19937 random(42);
    std::uniform_real_distribution<float> uniform(-0.9f, 0.9f);
    auto randomVector = [&] { return V(uniform(random), uniform(random), uniform(random)); };
    constexpr const char* names[] = {"sah", "lbvh", "tiny", "tiny_leaf1", "lbvh_radix"};
    constexpr int backendCount = 5;
    unsigned mismatches = 0;
    std::puts("scene,backend,threads,triangles,rays,bounces,coherent,build_ms,trace_ms,node_bytes,mismatches,checksum");
    for (bool clutter : {false, true}) {
        std::vector<bvh2::Primitive> triangles;
        for (int axis = 0; axis < 3; ++axis) for (float side : {-1.0f, 1.0f})
            for (int y = 0; y < grid; ++y) for (int x = 0; x < grid; ++x) {
                V p[4];
                for (int k = 0; k < 4; ++k) {
                    p[k][axis] = side;
                    p[k][(axis + 1) % 3] = -1 + 2.0f * (x + (k & 1)) / grid;
                    p[k][(axis + 2) % 3] = -1 + 2.0f * (y + (k >> 1)) / grid;
                }
                triangles.emplace_back(p[0], p[1], p[2]);
                triangles.emplace_back(p[2], p[1], p[3]);
            }
        if (clutter) for (int i = 0; i < grid * grid; ++i) {
            const V a = randomVector(), u = randomVector(), v = randomVector();
            triangles.emplace_back(a, a + V(u.x * .05f, u.y * .05f, u.z * .05f),
                                      a + V(v.x * .05f, v.y * .05f, v.z * .05f));
        }
        std::vector<tinybvh::bvhvec4> vertices;
        for (const auto& t : triangles) for (const auto& p : {t.v0, t.v1, t.v2})
            vertices.emplace_back(p.x, p.y, p.z, 0);
        bvh2::BVHData<float> nodes[3];
        std::unique_ptr<tinybvh::BVH> tiny[2];
        std::vector<double> buildTimes[backendCount];
        for (int round = 0; round < 4; ++round) for (int backend = 0; backend < backendCount; ++backend) {
            if (backend == 2 || backend == 3) tiny[backend - 2] = std::make_unique<tinybvh::BVH>();
            else nodes[backend == 4 ? 2 : backend] = bvh2::BVHData<float>{};
            const double ms = timed([&] {
                if (!backend) nodes[backend] = bvh2::buildSAHBVH(executor, triangles);
                else if (backend == 1) nodes[backend] = bvh2::buildLBVH<uint64_t>(executor, triangles);
                else if (backend == 4) nodes[2] = bvh2::buildLBVH<uint64_t>(executor, triangles, bvh2::MortonSortMethod::RadixSort);
                else {
                    auto& tree = *tiny[backend - 2];
                    tree.settings.enableThreading = workers > 1;
                    tree.Build(vertices.data(), uint32_t(triangles.size()));
                    if (backend == 3) { tree.Compact(); tree.SplitLeafs(1); tree.Refit(); }
                }
            });
            if (round) buildTimes[backend].push_back(ms);
        }
        for (bool coherent : {true, false}) {
            std::vector<bvh2::Ray> rays;
            for (int i = 0; i < rayCount; ++i) {
                const V origin = randomVector(), d = randomVector();
                rays.emplace_back(origin, unit(coherent ? V(d.x * .02f, d.y * .02f, 1) : d), 0, 100);
            }
            for (unsigned bounces : {1u, 8u}) {
                std::vector<Result> results[backendCount];
                for (auto& result : results) result.resize(rayCount);
                std::vector<double> times[backendCount];
                auto run = [&](auto&& closest, std::vector<Result>& output) {
                    auto traceRay = [&](int i) {
                        auto ray = rays[i];
                        Result result{ray.origin};
                        for (unsigned bounce = 0; bounce < bounces; ++bounce) {
                            uint32_t id; float t;
                            if (!closest(ray, id, t)) break;
                            const auto& tri = triangles[id];
                            V normal = unit(bvh2::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                            float projection = bvh2::dot(ray.direction, normal);
                            if (projection > 0) { normal = V(-normal.x, -normal.y, -normal.z); projection = -projection; }
                            V position;
                            for (int axis = 0; axis < 3; ++axis) {
                                position[axis] = ray.origin[axis] + t * ray.direction[axis];
                                ray.direction[axis] -= 2 * projection * normal[axis];
                                ray.origin[axis] = position[axis] + 1e-4f * normal[axis];
                            }
                            const V segment = position - result.end;
                            result.path += std::sqrt(bvh2::dot(segment, segment));
                            result.end = position;
                            ++result.hits;
                        }
                        output[i] = result;
                    };
                    if (workers == 1) {
                        for (int i = 0; i < rayCount; ++i) traceRay(i);
                    } else {
                        tf::Taskflow flow;
                        flow.for_each_index(0, rayCount, 1, traceRay);
                        executor.run(flow).get();
                    }
                };
                for (int round = 0; round < 6; ++round) for (int slot = 0; slot < backendCount; ++slot) {
                    const int backend = (slot + round) % backendCount;
                    auto trace = [&] {
                        if (backend < 2 || backend == 4) run([&](const bvh2::Ray& ray, uint32_t& id, float& t) {
                            return bvh2::traverseBVHClosestHit(nodes[backend == 4 ? 2 : backend], triangles, ray, id, t);
                        }, results[backend]);
                        else run([&](const bvh2::Ray& ray, uint32_t& id, float& t) {
                            tinybvh::Ray r({ray.origin.x, ray.origin.y, ray.origin.z},
                                           {ray.direction.x, ray.direction.y, ray.direction.z}, ray.tmax);
                            tiny[backend - 2]->Intersect(r);
                            id = r.hit.prim; t = r.hit.t;
                            return t < ray.tmax;
                        }, results[backend]);
                    };
                    if (round) times[backend].push_back(timed(trace));
                    else {
                        const auto until = Clock::now() + std::chrono::milliseconds(100);
                        do { trace(); } while (Clock::now() < until);
                    }
                }
                for (int backend = 0; backend < backendCount; ++backend) {
                    unsigned errors = 0;
                    double checksum = 0;
                    for (int i = 0; i < rayCount; ++i) {
                        const auto& a = results[0][i]; const auto& b = results[backend][i];
                        const V delta = a.end - b.end;
                        if (a.hits != b.hits || std::abs(a.path - b.path) > .005f ||
                            bvh2::dot(delta, delta) > .005f * .005f || !std::isfinite(b.path)) {
                            if (!errors) std::fprintf(stderr, "backend=%d ray=%d hits=%u/%u path=%g/%g\n",
                                backend, i, a.hits, b.hits, a.path, b.path);
                            ++errors;
                        }
                        checksum += b.path + b.hits;
                    }
                    const size_t bytes = (backend == 2 || backend == 3) ? tiny[backend - 2]->usedNodes * sizeof(tinybvh::BVH::BVHNode)
                                                     : nodes[backend == 4 ? 2 : backend].size() * (sizeof(bvh2::Box<float>) + 2 * sizeof(uint32_t));
                    std::printf("%s,%s,%u,%zu,%d,%u,%d,%.3f,%.3f,%zu,%u,%.6f\n", clutter ? "clutter" : "cube",
                        names[backend], workers, triangles.size(), rayCount,
                        bounces, coherent, median(buildTimes[backend]), median(times[backend]), bytes, errors, checksum);
                    mismatches += errors;
                }
            }
        }
    }
    return mismatches ? 1 : 0;
}
