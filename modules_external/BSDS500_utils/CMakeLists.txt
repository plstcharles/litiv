
project(BSDS500_utils)

file(GLOB SOURCE_FILES
    "src/*.cpp"
    "src/*.cc"
    "src/*.c"
)

file(GLOB INCLUDE_FILES
    "external/BSDS500_utils/*.hpp"
    "external/BSDS500_utils/*.hh"
    "external/BSDS500_utils/*.h"
)

add_library(BSDS500_utils SHARED ${SOURCE_FILES})

target_compil_litiv_dependencies(BSDS500_utils)
target_include_directories(BSDS500_utils
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)
target_link_litiv_dependencies(BSDS500_utils)

install(TARGETS BSDS500_utils DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
