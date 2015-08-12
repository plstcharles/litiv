
project(litiv_world)

file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/include/litiv/litiv_modules.hpp)
set(CMAKE_CONFIGURABLE_FILE_CONTENT "#pragma once\n")
foreach(l ${litiv_modules})
    set(CMAKE_CONFIGURABLE_FILE_CONTENT "${CMAKE_CONFIGURABLE_FILE_CONTENT}\n#include \"litiv/${l}.hpp\"")
endforeach(l ${${litiv_modules}})
set(CMAKE_CONFIGURABLE_FILE_CONTENT "${CMAKE_CONFIGURABLE_FILE_CONTENT}\n")
configure_file(
    "${CMAKE_ROOT}/Modules/CMakeConfigurableFile.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/litiv_modules.hpp"
)
unset(CMAKE_CONFIGURABLE_FILE_CONTENT)

file(GLOB SOURCE_FILES
    "src/*.cpp"
)

file(GLOB INCLUDE_FILES
    "litiv/*.hpp"
    "litiv/*.ipp"
)

add_library(litiv_world SHARED ${SOURCE_FILES})

target_link_libraries(litiv_world ${litiv_projects})
target_compil_litiv_dependencies(litiv_world)
target_include_directories(litiv_world
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)

install(TARGETS litiv_world DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
