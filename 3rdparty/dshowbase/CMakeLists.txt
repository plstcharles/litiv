
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(WIN32)

    litiv_3rdparty_module(dshowbase)

    add_files(SOURCE_FILES
        "src/wxlist.cpp"
        "src/wxutil.cpp"
        "src/amextra.cpp"
        "src/amfilter.cpp"
        "src/amvideo.cpp"
        "src/arithutil.cpp"
        "src/combase.cpp"
        "src/cprop.cpp"
        "src/ctlutil.cpp"
        "src/ddmm.cpp"
        "src/dllentry.cpp"
        "src/dllsetup.cpp"
        "src/mtype.cpp"
        "src/outputq.cpp"
        "src/perflog.cpp"
        "src/pstream.cpp"
        "src/pullpin.cpp"
        "src/refclock.cpp"
        "src/renbase.cpp"
        "src/schedule.cpp"
        "src/seekpt.cpp"
        "src/source.cpp"
        "src/strmctl.cpp"
        "src/sysclock.cpp"
        "src/transfrm.cpp"
        "src/transip.cpp"
        "src/videoctl.cpp"
        "src/vtrans.cpp"
        "src/winctrl.cpp"
        "src/winutil.cpp"
        "src/wxdebug.cpp"
    )
    add_files(INCLUDE_FILES
        "include/litiv/3rdparty/dshowbase/wxutil.h"
        "include/litiv/3rdparty/dshowbase/amextra.h"
        "include/litiv/3rdparty/dshowbase/amfilter.h"
        "include/litiv/3rdparty/dshowbase/cache.h"
        "include/litiv/3rdparty/dshowbase/checkbmi.h"
        "include/litiv/3rdparty/dshowbase/combase.h"
        "include/litiv/3rdparty/dshowbase/cprop.h"
        "include/litiv/3rdparty/dshowbase/ctlutil.h"
        "include/litiv/3rdparty/dshowbase/ddmm.h"
        "include/litiv/3rdparty/dshowbase/dllsetup.h"
        "include/litiv/3rdparty/dshowbase/dmodshowutil.h"
        "include/litiv/3rdparty/dshowbase/dshowutil.h"
        "include/litiv/3rdparty/dshowbase/dxmperf.h"
        "include/litiv/3rdparty/dshowbase/fourcc.h"
        "include/litiv/3rdparty/dshowbase/implcommon.h"
        "include/litiv/3rdparty/dshowbase/measure.h"
        "include/litiv/3rdparty/dshowbase/msgthrd.h"
        "include/litiv/3rdparty/dshowbase/mtype.h"
        "include/litiv/3rdparty/dshowbase/outputq.h"
        "include/litiv/3rdparty/dshowbase/perflog.h"
        "include/litiv/3rdparty/dshowbase/perfstruct.h"
        "include/litiv/3rdparty/dshowbase/pstream.h"
        "include/litiv/3rdparty/dshowbase/pullpin.h"
        "include/litiv/3rdparty/dshowbase/qedit.h"
        "include/litiv/3rdparty/dshowbase/refclock.h"
        "include/litiv/3rdparty/dshowbase/reftime.h"
        "include/litiv/3rdparty/dshowbase/renbase.h"
        "include/litiv/3rdparty/dshowbase/schedule.h"
        "include/litiv/3rdparty/dshowbase/seekpt.h"
        "include/litiv/3rdparty/dshowbase/source.h"
        "include/litiv/3rdparty/dshowbase/streams.h"
        "include/litiv/3rdparty/dshowbase/strmctl.h"
        "include/litiv/3rdparty/dshowbase/sysclock.h"
        "include/litiv/3rdparty/dshowbase/transfrm.h"
        "include/litiv/3rdparty/dshowbase/transip.h"
        "include/litiv/3rdparty/dshowbase/videoctl.h"
        "include/litiv/3rdparty/dshowbase/vtrans.h"
        "include/litiv/3rdparty/dshowbase/winctrl.h"
        "include/litiv/3rdparty/dshowbase/winutil.h"
        "include/litiv/3rdparty/dshowbase/wxdebug.h"
        "include/litiv/3rdparty/dshowbase/wxlist.h"
    )
    option(BUILD_DSHOWBASE_GMFBRIDGE "Specifies whether the GMFBridge utility should be included/linked in the dshowbase project build" OFF)
    if(BUILD_DSHOWBASE_GMFBRIDGE)
        add_files(SOURCE_FILES
            "src/GMF/source.cpp"
            "src/GMF/bridge.cpp"
            "src/GMF/cp.cpp"
            "src/GMF/GMFBridge_i.c"
            "src/GMF/GMFBridge_p.c"
            "src/GMF/sink.cpp"
        )
        add_files(INCLUDE_FILES
            "include/litiv/3rdparty/dshowbase/gmf/smartPtr.h"
            "include/litiv/3rdparty/dshowbase/gmf/source.h"
            "include/litiv/3rdparty/dshowbase/gmf/bridge.h"
            "include/litiv/3rdparty/dshowbase/gmf/cp.h"
            "include/litiv/3rdparty/dshowbase/gmf/dispatch.h"
            "include/litiv/3rdparty/dshowbase/gmf/GMFBridge_h.h"
            "include/litiv/3rdparty/dshowbase/gmf/sink.h"
        )
    endif()
    option(BUILD_DSHOWBASE_OCVCOMPAT "Specifies whether the OpenCV compat utility should be included/linked in the dshowbase project build" ON)
    if(BUILD_DSHOWBASE_OCVCOMPAT)
        add_files(SOURCE_FILES
            "src/ocvcompat.cpp"
        )
        add_files(INCLUDE_FILES
            "include/litiv/3rdparty/dshowbase/ocvcompat.h"
        )
    endif()

    add_library(litiv_dshowbase STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
    set_target_properties(litiv_dshowbase PROPERTIES FOLDER "3rdparty")
    target_include_directories(litiv_dshowbase
        PUBLIC
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
            "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
            "$<INSTALL_INTERFACE:include/>"
    )
    target_link_libraries(litiv_dshowbase
        strmiids
        winmm
        msdmo
        dmoguids
        wmcodecdspuuid
    )
    if(BUILD_DSHOWBASE_OCVCOMPAT)
        target_include_directories(litiv_dshowbase
            PUBLIC
                "$<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>"
        )
        target_link_libraries(litiv_dshowbase
            ${OpenCV_LIBRARIES}
        )
    endif()

    install(TARGETS litiv_dshowbase
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        COMPONENT 3rdparty
    )
    install(DIRECTORY "include/"
        DESTINATION "include/"
        COMPONENT 3rdparty
        FILES_MATCHING PATTERN "*.hpp"
    )
endif()
