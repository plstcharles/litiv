
project(edges)

include_directories(${litiv_imgproc_INCLUDE_DIR} ${litiv_features2d_INCLUDE_DIR} ${litiv_utils_INCLUDE_DIR})

add_executable(edges src/main.cpp)

target_link_libraries(edges litiv_world)
