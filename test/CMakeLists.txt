add_executable(integration_test main.cpp)
target_link_libraries(integration_test PRIVATE BVH2)

include(GoogleTest)

function(bvh2_add_gtest target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE GTest::gtest_main BVH2)
    gtest_discover_tests(${target})
endfunction()

bvh2_add_gtest(unit_test_radix_sort unit_test_radix_sort.cpp)
bvh2_add_gtest(unit_test_BVH unit_test_BVH.cpp)
bvh2_add_gtest(unit_test_octree unit_test_octree.cpp)
bvh2_add_gtest(unit_test_quadtree unit_test_quadtree.cpp)

bvh2_add_gtest(perf_test_BVH perf_test_BVH.cpp)
bvh2_add_gtest(perf_test_radix_sort perf_test_radix_sort.cpp)
bvh2_add_gtest(perf_test_octree perf_test_octree.cpp)
bvh2_add_gtest(perf_test_quadtree unit_test_quadtree_performance.cpp)

