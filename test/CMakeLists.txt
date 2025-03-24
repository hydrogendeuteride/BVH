add_executable(integration_test main.cpp)
target_link_libraries(integration_test PRIVATE BVH2)

add_executable(unit_test_radix_sort unit_test_radix_sort.cpp)
target_link_libraries(unit_test_radix_sort PRIVATE GTest::gtest_main BVH2)

include(GoogleTest)
gtest_discover_tests(unit_test_radix_sort)
