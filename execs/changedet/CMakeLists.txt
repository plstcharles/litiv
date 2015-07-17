
project(changedet)

include_directories(${litiv_video_INCLUDE_DIR} ${litiv_features2d_INCLUDE_DIR} ${litiv_utils_INCLUDE_DIR})

add_executable(changedet src/main.cpp)

#if using glsl
target_link_libraries(changedet ${GLFW_LIBRARIES})
target_link_libraries(changedet ${OPENGL_LIBRARIES})
target_link_libraries(changedet ${GLEW_LIBRARIES})
target_link_libraries(changedet ${GLM_LIBRARIES})
#endif using glsl
target_link_libraries(changedet litiv_utils litiv_features2d litiv_video)
target_link_libraries(changedet ${OpenCV_LIBRARIES})
