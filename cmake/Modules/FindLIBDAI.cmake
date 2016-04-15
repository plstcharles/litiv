# Copyright (C) 2013 Bjoern Andres, Thorsten Beier and Joerg H.~Kappes.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# FindLIBDAI.cmake
#

FIND_PATH(LIBDAI_INCLUDE_DIR
  dai/alldai.h
  HINTS "~/usr/local/include/dai"
   "~/usr/include/dai"
)

FIND_LIBRARY(LIBDAI_LIBRARY
   libdai
   HINTS "~/usr/local/lib/dai"
   "~/usr/lib/dai"
)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LIBDAI
   DEFAULT_MSG
   LIBDAI_LIBRARY
   LIBDAI_INCLUDE_DIR
)

IF(LIBDAI_FOUND)

ELSE()
   SET(LIBDAI_INCLUDE_DIR LIBDAI_INCLUDE_DIR-NOTFOUND)
   SET(LIBDAI_LIBRARY   LIBDAI_LIBRARY-NOTFOUND)
ENDIF(LIBDAI_FOUND)
