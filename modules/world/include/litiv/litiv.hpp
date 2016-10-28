
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
//
/////////////////////////////////////////////////////////////////////////////
//
// This file is the main entrypoint used to include all LITIV modules into a
// project at once; it works a little bit like OpenCV's "world" module. In a
// similar way, you can link to all module libs via the "litiv_world" target.
//
/////////////////////////////////////////////////////////////////////////////

#pragma once

#include "litiv/litiv_modules.hpp"

namespace lv {

    /// for now, always returns true
    bool initAll();

} // namespace lv
