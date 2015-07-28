
project(litiv_imgproc)

file(GLOB SOURCE_FILES
    "src/*.cpp"
)

file(GLOB INCLUDE_FILES
    "litiv/imgproc/*.hpp"
    "litiv/imgproc/*.ipp"
)

add_library(litiv_imgproc SHARED ${SOURCE_FILES})

target_link_libraries(litiv_imgproc litiv_features2d)
target_compil_litiv_dependencies(litiv_imgproc)
target_include_directories(litiv_imgproc
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)

install(TARGETS litiv_imgproc DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
