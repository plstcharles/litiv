
project(BSDS500)

add_files(SOURCE_FILES
    "src/kofn.cpp"
    "src/csa.cpp"
)

add_files(INCLUDE_FILES
    "include/3rdparty/BSDS500/csa.hpp"
    "include/3rdparty/BSDS500/csa_defs.hpp"
    "include/3rdparty/BSDS500/csa_types.hpp"
    "include/3rdparty/BSDS500/kofn.hpp"
    "include/3rdparty/BSDS500/match.hpp"
)

add_library(BSDS500 STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

target_include_directories(BSDS500
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)
set_target_properties(BSDS500 PROPERTIES FOLDER "3rdparty")
#install(TARGETS BSDS500 DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
