
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2015 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "litiv/utils/cxx.hpp"

namespace lv {

    /// returns the executable's current working directory path; relies on getcwd, and may return an empty string
    std::string getCurrentWorkDirPath();
    /// adds a forward slash to the given directory path if it ends without one, with handling for special cases (useful for path concatenation)
    std::string addDirSlashIfMissing(const std::string& sDirPath);
    /// returns a sorted list of all files located at a given directory path
    std::vector<std::string> getFilesFromDir(const std::string& sDirPath);
    /// returns a sorted list of all subdirectories located at a given directory path
    std::vector<std::string> getSubDirsFromDir(const std::string& sDirPath);
    /// filters a list of paths using string tokens; if a token is found in a path, it is removed/kept from the list
    void filterFilePaths(std::vector<std::string>& vsFilePaths, const std::vector<std::string>& vsRemoveTokens, const std::vector<std::string>& vsKeepTokens);
    /// returns whether a local file or directory already exists
    bool checkIfExists(const std::string& sPath);
    /// creates a local directory at the given path if one does not already exist (does not work recursively)
    bool createDirIfNotExist(const std::string& sDirPath);
    /// creates a binary file at the specified location, and fills it with unspecified/zero data bytes (useful for critical/real-time stream writing without continuous reallocation)
    std::fstream createBinFileWithPrealloc(const std::string& sFilePath, size_t nPreallocBytes, bool bZeroInit=false);
    /// registers the SIGINT, SIGTERM, and SIGBREAK (if available) console signals to the given handler
    void registerAllConsoleSignals(void(*lHandler)(int));
    /// returns the amount of physical memory currently used on the system
    size_t getCurrentPhysMemBytesUsed();

} // namespace lv

#if defined(_MSC_VER)
namespace {
    template<class T>
    inline void SafeRelease(T** ppT) {
        if(*ppT) {
            (*ppT)->Release();
            *ppT = nullptr;
        }
    }
}
#endif //defined(_MSC_VER)