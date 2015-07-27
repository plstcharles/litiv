
project(litiv_video)

file(GLOB SOURCE_FILES
    "src/*.cpp"
)

file(GLOB INCLUDE_FILES
    "litiv/video/*.hpp"
    "litiv/video/*.ipp"
)

add_library(litiv_video SHARED ${SOURCE_FILES})

target_link_libraries(litiv_video litiv_features2d)
target_compil_litiv_dependencies(litiv_video)
target_include_directories(litiv_video
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)

install(TARGETS litiv_video DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
