
project(litiv_features2d)

file(GLOB SOURCE_FILES
    "src/*.cpp"
)

file(GLOB INCLUDE_FILES
    "litiv/features2d/*.hpp"
    "litiv/features2d/*.ipp"
)

add_library(litiv_features2d ${SOURCE_FILES})

target_link_libraries(litiv_features2d litiv_utils)
target_compil_litiv_dependencies(litiv_features2d)
target_include_directories(litiv_features2d
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)

install(TARGETS litiv_features2d DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
