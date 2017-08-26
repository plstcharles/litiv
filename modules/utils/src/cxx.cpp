
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

#include "litiv/utils/cxx.hpp"

std::string lv::putf(const char* acFormat, ...) {
    va_list vArgs;
    va_start(vArgs,acFormat);
    std::string vBuffer(1024,'\0');
#ifdef _DEBUG
    if(((&vBuffer[0])+vBuffer.size()-1)!=&vBuffer[vBuffer.size()-1])
        lvStdError_(runtime_error,"basic_string should have contiguous memory (need C++11!)");
#endif //defined(_DEBUG)
    const int nWritten = vsnprintf(&vBuffer[0],(int)vBuffer.size(),acFormat,vArgs);
    va_end(vArgs);
    if(nWritten<0)
        lvStdError_(runtime_error,"putf failed (1)");
    if((size_t)nWritten<=vBuffer.size()) {
        vBuffer.resize((size_t)nWritten);
        return vBuffer;
    }
    vBuffer.resize((size_t)nWritten+1);
    va_list vArgs2;
    va_start(vArgs2,acFormat);
    const int nWritten2 = vsnprintf(&vBuffer[0],(int)vBuffer.size(),acFormat,vArgs2);
    va_end(vArgs2);
    if(nWritten2<0 || (size_t)nWritten2>vBuffer.size())
        lvStdError_(runtime_error,"putf failed (2)");
    vBuffer.resize((size_t)nWritten2);
    return vBuffer;
}

bool lv::compare_lowercase(const std::string& i, const std::string& j) {
    std::string i_lower(i), j_lower(j);
    std::transform(i_lower.begin(),i_lower.end(),i_lower.begin(),[](auto c){return std::tolower(c);});
    std::transform(j_lower.begin(),j_lower.end(),j_lower.begin(),[](auto c){return std::tolower(c);});
    return i_lower<j_lower;
}

bool lv::string_contains_token(const std::string& s, const std::vector<std::string>& tokens) {
    for(size_t i=0; i<tokens.size(); ++i)
        if(s.find(tokens[i])!=std::string::npos)
            return true;
    return false;
}

std::string lv::clampString(const std::string& sInput, size_t nSize, char cPadding) {
    return sInput.size()>nSize?sInput.substr(0,nSize):std::string(nSize-sInput.size(),cPadding)+sInput;
}

std::vector<std::string> lv::split(const std::string& sInputStr, char cDelim) {
    std::vector<std::string> vsTokens;
    lv::split(sInputStr,std::back_inserter(vsTokens),cDelim);
    return vsTokens;
}

std::string lv::getTimeStamp() {
    std::time_t tNow = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    char acBuffer[128];
    std::strftime(acBuffer,sizeof(acBuffer),"%F %T",std::localtime(&tNow)); // std::put_time missing w/ GCC<5.0
    return std::string(acBuffer);
}

std::string lv::getVersionStamp() {
    return "LITIV Framework v" LITIV_VERSION_STR " (SHA1=" LITIV_VERSION_SHA1 ")";
}

std::string lv::getLogStamp() {
    return std::string("\n")+lv::getVersionStamp()+"\n["+lv::getTimeStamp()+"]\n";
}

std::mutex g_oPrintMutex;

std::mutex& lv::getLogMutex() {
    return g_oPrintMutex;
}

std::ostream& lv::safe_print(std::ostream& os, const char* acFormat, ...) {
    va_list vArgs;
    va_start(vArgs,acFormat);
    std::string vBuffer(1024,'\0');
#ifdef _DEBUG
    if(((&vBuffer[0])+vBuffer.size()-1)!=&vBuffer[vBuffer.size()-1])
        lvStdError_(runtime_error,"basic_string should have contiguous memory (need C++11!)");
#endif //defined(_DEBUG)
    const int nWritten = vsnprintf(&vBuffer[0],(int)vBuffer.size(),acFormat,vArgs);
    va_end(vArgs);
    if(nWritten<0)
        lvStdError_(runtime_error,"safe_print failed (1)");
    if((size_t)nWritten<=vBuffer.size()) {
        vBuffer.resize((size_t)nWritten);
        std::lock_guard<std::mutex> oLock(getLogMutex());
        return os << vBuffer;
    }
    vBuffer.resize((size_t)nWritten+1);
    va_list vArgs2;
    va_start(vArgs2,acFormat);
    const int nWritten2 = vsnprintf(&vBuffer[0],(int)vBuffer.size(),acFormat,vArgs2);
    va_end(vArgs2);
    if(nWritten2<0 || (size_t)nWritten2>vBuffer.size())
        lvStdError_(runtime_error,"safe_print failed (2)");
    vBuffer.resize((size_t)nWritten2);
    std::lock_guard<std::mutex> oLock(getLogMutex());
    return os << vBuffer;
}

int g_nVerbosity = 1;

int lv::getVerbosity() {
    return g_nVerbosity;
}

void lv::setVerbosity(int nLevel) {
    g_nVerbosity = nLevel;
}

void lv::doNotOptimizeCharPointer(char const volatile*) {}