
# This file is part of the LITIV framework; visit the original repository at
# https://github.com/plstcharles/litiv for more information.
#
# Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

find_package(Doxygen)
if(DOXYGEN_FOUND)
    option(BUILD_DOCUMENTATION "Generate HTML documentation (requires Doxygen)" ON)
    if(BUILD_DOCUMENTATION)
        find_package(LATEX COMPONENTS PDFLATEX)
        option(BUILD_LATEX_DOCUMENTATION "Generate PDF documentation along with HTML (requires LaTeX)" ${LATEX_FOUND})
        if(${BUILD_LATEX_DOCUMENTATION})
            set(DOXYGEN_GENERATE_LATEX "YES")
        else()
            set(DOXYGEN_GENERATE_LATEX "NO")
        endif()
        set(DOXYGEN_OUTPUT_PATH "doxygen")
        set(DOXYGEN_PROJECT_LOGO "${CMAKE_CURRENT_SOURCE_DIR}/images/litiv.ico")
        set(DOXYGEN_HTML_EXTRA_STYLESHEET)
        set(DOXYGEN_HTML_EXTRA_FILES)
        set(DOXYGEN_EXTRA_BIB_FILES "${CMAKE_CURRENT_SOURCE_DIR}/IEEEabrv.bib ${CMAKE_CURRENT_SOURCE_DIR}/litiv.bib ${paths_bib}")
        set(doxyfile_in "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in")
        set(doxyfile    "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile")
        #set(rootfile_in "${CMAKE_CURRENT_SOURCE_DIR}/root.markdown.in")
        #set(rootfile    "${CMAKE_CURRENT_BINARY_DIR}/root.markdown")
        set(paths_include)
        set(paths_doc)
        set(paths_bib)
        set(paths_sample)
        set(deps)
        foreach(m ${litiv_modules})
            set(include_dir "${CMAKE_SOURCE_DIR}/modules/${m}/include")
            if(EXISTS "${include_dir}")
                if(EXISTS "${CMAKE_SOURCE_DIR}/modules/${m}/include")
                    list(APPEND paths_include "${CMAKE_SOURCE_DIR}/modules/${m}/include")
                    list(APPEND deps "${CMAKE_SOURCE_DIR}/modules/${m}/include")
                endif()
                if(EXISTS "${CMAKE_BINARY_DIR}/modules/${m}/include")
                    list(APPEND paths_include "${CMAKE_BINARY_DIR}/modules/${m}/include")
                    list(APPEND deps "${CMAKE_BINARY_DIR}/modules/${m}/include")
                endif()
            endif()
            set(doc_dir "${CMAKE_SOURCE_DIR}/modules/${m}/doc")
            if(EXISTS "${doc_dir}")
                list(APPEND paths_doc "${doc_dir}")
                list(APPEND deps ${doc_dir})
                set(bib_file "${doc_dir}/${m}.bib")
                if(EXISTS "${bib_file}")
                    set(paths_bib "${paths_bib} ${bib_file}")
                    list(APPEND deps ${bib_file})
                endif()
            endif()
            set(sample_dir "${CMAKE_SOURCE_DIR}/modules/${m}/samples")
            if(EXISTS "${sample_dir}")
                list(APPEND paths_sample "${sample_dir}")
                list(APPEND deps ${sample_dir})
            endif()
        endforeach()
        list(APPEND deps ${CMAKE_CURRENT_SOURCE_DIR}/images ${CMAKE_SOURCE_DIR}/samples ${CMAKE_CURRENT_SOURCE_DIR}/IEEEabrv.bib ${CMAKE_CURRENT_SOURCE_DIR}/litiv.bib)
        list(APPEND DOXYGEN_HTML_EXTRA_FILES ${DOXYGEN_PROJECT_LOGO})
        string(REPLACE ";" " \\\n" DOXYGEN_HTML_EXTRA_FILES "${DOXYGEN_HTML_EXTRA_FILES}")
        #string(REPLACE ";" " \\\n" DOXYGEN_INPUT_LIST  "${rootfile} ; ${paths_include} ; ${paths_doc}")
        string(REPLACE ";" " \\\n" DOXYGEN_INPUT_LIST  "${paths_include} ; ${paths_doc}")
        string(REPLACE ";" " \\\n" DOXYGEN_IMAGE_PATH  "${CMAKE_CURRENT_SOURCE_DIR}/images ; ${paths_doc}")
        string(REPLACE ";" " \\\n" DOXYGEN_SAMPLE_PATH "${CMAKE_SOURCE_DIR}/samples ; ${paths_sample}")
        #configure_file(${rootfile_in} ${rootfile} @ONLY)
        configure_file(${doxyfile_in} ${doxyfile} @ONLY)
        add_custom_target(doc
            COMMAND
                ${DOXYGEN_EXECUTABLE} ${doxyfile}
            WORKING_DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS
                ${doxyfile}
                #${rootfile}
                ${deps}
            COMMENT
                "Generating API documentation..."
            VERBATIM
        )
        install(
            DIRECTORY
                ${CMAKE_CURRENT_BINARY_DIR}/doxygen/html
                ${CMAKE_CURRENT_BINARY_DIR}/doxygen/latex
            DESTINATION
                share/doc/litiv/
            COMPONENT
                docs
            OPTIONAL
        )
    endif()
endif()
