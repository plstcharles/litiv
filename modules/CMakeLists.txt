
get_subdirectory_list(subdirs ${CMAKE_CURRENT_SOURCE_DIR})
foreach(subdir ${subdirs})
    if(NOT(${subdir} STREQUAL "world"))
        add_subdirectory(${subdir})
    endif(NOT(${subdir} STREQUAL "world"))
endforeach()
add_subdirectory("world")
