
litiv_module(utils)

file(REMOVE ${CMAKE_CURRENT_BINARY_DIR}/include/litiv/${LITIV_CURRENT_MODULE_NAME}/DefineUtils.hpp)
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/include/litiv/${LITIV_CURRENT_MODULE_NAME}/DefineUtils.hpp.in"
    "${CMAKE_CURRENT_BINARY_DIR}/include/litiv/${LITIV_CURRENT_MODULE_NAME}/DefineUtils.hpp"
)

file(GLOB SOURCE_FILES
    "src/*.cpp"
)

file(GLOB INCLUDE_FILES
    "litiv/${LITIV_CURRENT_MODULE_NAME}/*.hpp"
    "litiv/${LITIV_CURRENT_MODULE_NAME}/*.ipp"
)

add_library(${LITIV_CURRENT_PROJECT_NAME} STATIC ${SOURCE_FILES})

if(USE_BSDS500_BENCHMARK)
    target_link_libraries(${LITIV_CURRENT_PROJECT_NAME} BSDS500)
endif(USE_BSDS500_BENCHMARK)
target_link_litiv_dependencies(${LITIV_CURRENT_PROJECT_NAME})
target_compil_litiv_dependencies(${LITIV_CURRENT_PROJECT_NAME})
target_include_directories(${LITIV_CURRENT_PROJECT_NAME}
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
    PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
    INTERFACE "$<INSTALL_INTERFACE:include/>"
)

#install(TARGETS ${LITIV_CURRENT_PROJECT_NAME} DESTINATION lib)
install(FILES ${INCLUDE_FILES} DESTINATION include)
