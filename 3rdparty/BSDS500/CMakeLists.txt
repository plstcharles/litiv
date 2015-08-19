
project(BSDS500)

file(GLOB SOURCE_FILES
    "src/*.cpp"
    "src/*.cc"
    "src/*.c"
)

file(GLOB INCLUDE_FILES
    "3rdparty/BSDS500/*.hpp"
    "3rdparty/BSDS500/*.hh"
    "3rdparty/BSDS500/*.h"
)

add_library(BSDS500 STATIC ${SOURCE_FILES})

target_include_directories(BSDS500
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)

#install(TARGETS BSDS500 DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
