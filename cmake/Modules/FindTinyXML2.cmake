
find_path(TinyXML2_INCLUDE_DIRS
    NAMES
        tinyxml2.h
    HINTS
        ${TINYXML2_ROOT_DIR}
        "$ENV{USER_DEVELOP}/tinyxml2/"
        "$ENV{USER_DEVELOP}/tinyxml2/include/"
        "$ENV{USER_DEVELOP}/vendor/tinyxml2/"
        "$ENV{USER_DEVELOP}/vendor/tinyxml2/include/"
)

find_library(TinyXML2_LIBRARIES
    NAMES
        tinyxml2
    HINTS
        ${TINYXML2_ROOT_DIR}/build/Release/
        "$ENV{USER_DEVELOP}/tinyxml2/build/Release/"
        "$ENV{USER_DEVELOP}/vendor/tinyxml2/build/Release/"
)

set(TinyXML2_INCLUDE_DIR ${TinyXML2_INCLUDE_DIRS})
set(TinyXML2_LIBRARY ${TinyXML2_LIBRARIES})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(TinyXML2
    REQUIRED_VARS
        TinyXML2_INCLUDE_DIRS
        TinyXML2_LIBRARIES
)

mark_as_advanced(
  TinyXML2_INCLUDE_DIRS
  TinyXML2_LIBRARIES
)
