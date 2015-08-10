
project(litiv_utils)

file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/DefineUtils.hpp)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/utils/DefineUtils.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/utils/DefineUtils.hpp"
)

file(GLOB SOURCE_FILES
    "src/*.cpp"
)

file(GLOB INCLUDE_FILES
    "litiv/utils/*.hpp"
    "litiv/utils/*.ipp"
)

add_library(litiv_utils SHARED ${SOURCE_FILES})

if(USE_BSDS500_BENCHMARK_UTILS)
    target_link_libraries(litiv_utils BSDS500_utils)
endif(USE_BSDS500_BENCHMARK_UTILS)
target_compil_litiv_dependencies(litiv_utils)
target_include_directories(litiv_utils
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)
target_link_litiv_dependencies(litiv_utils)

install(TARGETS litiv_utils DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
