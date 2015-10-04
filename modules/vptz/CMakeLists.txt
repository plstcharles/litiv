
option(USE_VPTZ_STANDALONE "Build VPTZ library as standalone lib from LITIV framework" ON)

if(USE_VPTZ_STANDALONE)
    project(litiv_vptz)
    set(LITIV_CURRENT_MODULE_NAME vptz)
    set(LITIV_CURRENT_PROJECT_NAME litiv_vptz)
else(NOT USE_VPTZ_STANDALONE)
    litiv_module(vptz)
endif(USE_VPTZ_STANDALONE)

if(NOT ${USE_GLSL})
    message(ERROR "VirtualPTZ library requires full OpenGL support")
endif(NOT ${USE_GLSL})

file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/include/litiv/${LITIV_CURRENT_MODULE_NAME}/StandaloneUtils.hpp)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/${LITIV_CURRENT_MODULE_NAME}/StandaloneUtils.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/${LITIV_CURRENT_MODULE_NAME}/StandaloneUtils.hpp"
)

file(GLOB SOURCE_FILES
    "src/*.cpp"
)

file(GLOB INCLUDE_FILES
    "litiv/${LITIV_CURRENT_MODULE_NAME}/*.hpp"
    "litiv/${LITIV_CURRENT_MODULE_NAME}/*.ipp"
)

if(USE_VPTZ_STANDALONE)
    add_library(${LITIV_CURRENT_PROJECT_NAME} SHARED ${SOURCE_FILES})
    target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} ${OpenCV_LIBRARIES})
    if(USE_GLFW)
        target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} ${GLFW_LIBRARIES})
    elseif(USE_GLUT)
        target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} ${GLUT_LIBRARIES})
    endif(USE_GLFW)
    target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} ${OPENGL_LIBRARIES})
    target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} ${GLEW_LIBRARIES})
else(NOT USE_VPTZ_STANDALONE)
    add_library(${LITIV_CURRENT_PROJECT_NAME} STATIC ${SOURCE_FILES})
    target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} litiv_utils)
endif(USE_VPTZ_STANDALONE)
target_include_directories(${LITIV_CURRENT_PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)

#install(TARGETS ${LITIV_CURRENT_PROJECT_NAME} DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
