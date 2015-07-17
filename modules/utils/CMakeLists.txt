
project(litiv_utils)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/utils/DefineUtils.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/DefineUtils.hpp"
)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/utils/ParallelUtils.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/ParallelUtils.hpp"
)

file(GLOB SOURCE_FILES
    "src/*.cpp"
)

file(GLOB INCLUDE_FILES
    "litiv/utils/*.hpp"
    "litiv/utils/*.ipp"
)

add_library(litiv_utils ${SOURCE_FILES})

target_compil_litiv_dependencies(litiv_utils)
target_include_directories(litiv_utils
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)

install(TARGETS litiv_utils DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
