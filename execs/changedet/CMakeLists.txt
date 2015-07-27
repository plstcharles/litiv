
project(changedet)

include_directories(${litiv_video_INCLUDE_DIR} ${litiv_features2d_INCLUDE_DIR} ${litiv_utils_INCLUDE_DIR})

add_executable(changedet src/main.cpp)

target_compil_litiv_dependencies(changedet)
target_link_libraries(changedet litiv_utils litiv_features2d litiv_video)
