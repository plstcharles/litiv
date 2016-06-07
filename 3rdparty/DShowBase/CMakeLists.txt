
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
    project(DShowBase)
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
        "src/ocvcompat.cpp"
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
        "include/litiv/3rdparty/DShowBase/wxutil.h"
        "include/litiv/3rdparty/DShowBase/amextra.h"
        "include/litiv/3rdparty/DShowBase/amfilter.h"
        "include/litiv/3rdparty/DShowBase/cache.h"
        "include/litiv/3rdparty/DShowBase/checkbmi.h"
        "include/litiv/3rdparty/DShowBase/combase.h"
        "include/litiv/3rdparty/DShowBase/cprop.h"
        "include/litiv/3rdparty/DShowBase/ctlutil.h"
        "include/litiv/3rdparty/DShowBase/ddmm.h"
        "include/litiv/3rdparty/DShowBase/dllsetup.h"
        "include/litiv/3rdparty/DShowBase/dmodshowutil.h"
        "include/litiv/3rdparty/DShowBase/dshowutil.h"
        "include/litiv/3rdparty/DShowBase/dxmperf.h"
        "include/litiv/3rdparty/DShowBase/fourcc.h"
        "include/litiv/3rdparty/DShowBase/implcommon.h"
        "include/litiv/3rdparty/DShowBase/measure.h"
        "include/litiv/3rdparty/DShowBase/msgthrd.h"
        "include/litiv/3rdparty/DShowBase/mtype.h"
        "include/litiv/3rdparty/DShowBase/outputq.h"
        "include/litiv/3rdparty/DShowBase/ocvcompat.h"
        "include/litiv/3rdparty/DShowBase/perflog.h"
        "include/litiv/3rdparty/DShowBase/perfstruct.h"
        "include/litiv/3rdparty/DShowBase/pstream.h"
        "include/litiv/3rdparty/DShowBase/pullpin.h"
        "include/litiv/3rdparty/DShowBase/qedit.h"
        "include/litiv/3rdparty/DShowBase/refclock.h"
        "include/litiv/3rdparty/DShowBase/reftime.h"
        "include/litiv/3rdparty/DShowBase/renbase.h"
        "include/litiv/3rdparty/DShowBase/schedule.h"
        "include/litiv/3rdparty/DShowBase/seekpt.h"
        "include/litiv/3rdparty/DShowBase/source.h"
        "include/litiv/3rdparty/DShowBase/streams.h"
        "include/litiv/3rdparty/DShowBase/strmctl.h"
        "include/litiv/3rdparty/DShowBase/sysclock.h"
        "include/litiv/3rdparty/DShowBase/transfrm.h"
        "include/litiv/3rdparty/DShowBase/transip.h"
        "include/litiv/3rdparty/DShowBase/videoctl.h"
        "include/litiv/3rdparty/DShowBase/vtrans.h"
        "include/litiv/3rdparty/DShowBase/winctrl.h"
        "include/litiv/3rdparty/DShowBase/winutil.h"
        "include/litiv/3rdparty/DShowBase/wxdebug.h"
        "include/litiv/3rdparty/DShowBase/wxlist.h"
    )
    option(BUILD_DSHOWBASE_GMFBRIDGE "Sets whether the GMFBridge utility should be included in the DShowBase project build" OFF)
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
            "include/litiv/3rdparty/DShowBase/GMF/smartPtr.h"
            "include/litiv/3rdparty/DShowBase/GMF/source.h"
            "include/litiv/3rdparty/DShowBase/GMF/bridge.h"
            "include/litiv/3rdparty/DShowBase/GMF/cp.h"
            "include/litiv/3rdparty/DShowBase/GMF/dispatch.h"
            "include/litiv/3rdparty/DShowBase/GMF/GMFBridge_h.h"
            "include/litiv/3rdparty/DShowBase/GMF/sink.h"
        )
    endif()

    add_library(DShowBase STATIC ${SOURCE_FILES} ${INCLUDE_FILES})
    target_link_libraries(DShowBase strmiids winmm msdmo dmoguids wmcodecdspuuid)
    target_include_directories(DShowBase
        PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>"
        PUBLIC "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>"
        INTERFACE "$<INSTALL_INTERFACE:include/>"
    )
    set_target_properties(DShowBase PROPERTIES FOLDER "3rdparty")
    install(TARGETS DShowBase
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