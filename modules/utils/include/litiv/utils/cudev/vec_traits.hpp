
// This file is part of the LITIV framework; visit the original repository at
// https://github.com/plstcharles/litiv for more information.
//
// Copyright 2016 Pierre-Luc St-Charles; pierre-luc.st-charles<at>polymtl.ca
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

#include "litiv/utils/cudev/common.hpp"

namespace lv { namespace cudev {

    /// specialized version of cv::cudev's VecTraits to add support for vector reduction & part-based construction
    template<typename T> struct VecTraits;

    #define LV_CUDEV_VEC_TRAITS_INST(type,reducttype) \
        template <> struct VecTraits<type> \
        { \
            typedef type elem_type; \
            typedef reducttype reduct_elem_type; \
            enum {cn=1}; \
            __host__ __device__ __forceinline__ static type all(type v) {return v;} \
            __host__ __device__ __forceinline__ static type make(type x) {return x;} \
            __host__ __device__ __forceinline__ static type make(const type* v) {return *v;} \
            __host__ __device__ __forceinline__ static type make(type ## 1 v) {return v.x;} \
            __host__ __device__ __forceinline__ static reduct_elem_type reduce(const type v) {return reduct_elem_type(v);} \
        }; \
        template <> struct VecTraits<type ## 1> \
        { \
            typedef type elem_type; \
            typedef reducttype reduct_elem_type; \
            enum {cn=1}; \
            __host__ __device__ __forceinline__ static type ## 1 all(type v) {return make_ ## type ## 1(v);} \
            __host__ __device__ __forceinline__ static type ## 1 make(type x) {return make_ ## type ## 1(x);} \
            __host__ __device__ __forceinline__ static type ## 1 make(const type* v) {return make_ ## type ## 1(*v);} \
            __host__ __device__ __forceinline__ static reduct_elem_type reduce(const type ## 1 v) {return reduct_elem_type(v.x);} \
        }; \
        template <> struct VecTraits<type ## 2> \
        { \
            typedef type elem_type; \
            typedef reducttype reduct_elem_type; \
            enum {cn=2}; \
            __host__ __device__ __forceinline__ static type ## 2 all(type v) {return make_ ## type ## 2(v, v);} \
            __host__ __device__ __forceinline__ static type ## 2 make(type x, type y) {return make_ ## type ## 2(x, y);} \
            __host__ __device__ __forceinline__ static type ## 2 make(const type* v) {return make_ ## type ## 2(v[0], v[1]);} \
            __host__ __device__ __forceinline__ static type ## 2 make(type ## 1 v, type y=0) {return make_ ## type ## 2(v.x, y);} \
            __host__ __device__ __forceinline__ static reduct_elem_type reduce(const type ## 2 v) {return reduct_elem_type(v.x)+v.y;} \
        }; \
        template <> struct VecTraits<type ## 3> \
        { \
            typedef type elem_type; \
            typedef reducttype reduct_elem_type; \
            enum {cn=3}; \
            __host__ __device__ __forceinline__ static type ## 3 all(type v) {return make_ ## type ## 3(v, v, v);} \
            __host__ __device__ __forceinline__ static type ## 3 make(type x, type y, type z) {return make_ ## type ## 3(x, y, z);} \
            __host__ __device__ __forceinline__ static type ## 3 make(const type* v) {return make_ ## type ## 3(v[0], v[1], v[2]);} \
            __host__ __device__ __forceinline__ static type ## 3 make(type ## 2 v, type z=0) {return make_ ## type ## 3(v.x, v.y, z);} \
            __host__ __device__ __forceinline__ static type ## 3 make(type ## 1 v, type y=0, type z=0) {return make_ ## type ## 3(v.x, y, z);} \
            __host__ __device__ __forceinline__ static reduct_elem_type reduce(const type ## 3 v) {return reduct_elem_type(v.x)+v.y+v.z;} \
        }; \
        template <> struct VecTraits<type ## 4> \
        { \
            typedef type elem_type; \
            typedef reducttype reduct_elem_type; \
            enum {cn=4}; \
            __host__ __device__ __forceinline__ static type ## 4 all(type v) {return make_ ## type ## 4(v, v, v, v);} \
            __host__ __device__ __forceinline__ static type ## 4 make(type x, type y, type z, type w) {return make_ ## type ## 4(x, y, z, w);} \
            __host__ __device__ __forceinline__ static type ## 4 make(const type* v) {return make_ ## type ## 4(v[0], v[1], v[2], v[3]);} \
            __host__ __device__ __forceinline__ static type ## 4 make(type ## 3 v, type w=0) {return make_ ## type ## 4(v.x, v.y, v.z, w);} \
            __host__ __device__ __forceinline__ static type ## 4 make(type ## 2 v, type z=0, type w=0) {return make_ ## type ## 4(v.x, v.y, z, w);} \
            __host__ __device__ __forceinline__ static type ## 4 make(type ## 1 v, type y=0, type z=0, type w=0) {return make_ ## type ## 4(v.x, y, z, w);} \
            __host__ __device__ __forceinline__ static reduct_elem_type reduce(const type ## 4 v) {return reduct_elem_type(v.x)+v.y+v.z+v.w;} \
        };

    LV_CUDEV_VEC_TRAITS_INST(uchar, int)
    LV_CUDEV_VEC_TRAITS_INST(ushort, int)
    LV_CUDEV_VEC_TRAITS_INST(short, int)
    LV_CUDEV_VEC_TRAITS_INST(int, int)
    LV_CUDEV_VEC_TRAITS_INST(uint, uint)
    LV_CUDEV_VEC_TRAITS_INST(float, float)
    LV_CUDEV_VEC_TRAITS_INST(double, double)

    #undef LV_CUDEV_VEC_TRAITS_INST

    template<> struct VecTraits<schar>
    {
        typedef schar elem_type;
        typedef int reduct_elem_type;
        enum {cn=1};
        __host__ __device__ __forceinline__ static schar all(schar v) {return v;}
        __host__ __device__ __forceinline__ static schar make(schar x) {return x;}
        __host__ __device__ __forceinline__ static schar make(const schar* x) {return *x;}
        __host__ __device__ __forceinline__ static schar make(char1 v) {return v.x;}
        __host__ __device__ __forceinline__ static reduct_elem_type reduce(const schar v) {return reduct_elem_type(v);}
    };
    template<> struct VecTraits<char1>
    {
        typedef schar elem_type;
        typedef int reduct_elem_type;
        enum {cn=1};
        __host__ __device__ __forceinline__ static char1 all(schar v) {return make_char1(v);}
        __host__ __device__ __forceinline__ static char1 make(schar x) {return make_char1(x);}
        __host__ __device__ __forceinline__ static char1 make(const schar* v) {return make_char1(v[0]);}
        __host__ __device__ __forceinline__ static reduct_elem_type reduce(const char1 v) {return reduct_elem_type(v.x);}
    };
    template<> struct VecTraits<char2>
    {
        typedef schar elem_type;
        typedef int reduct_elem_type;
        enum {cn=2};
        __host__ __device__ __forceinline__ static char2 all(schar v) {return make_char2(v, v);}
        __host__ __device__ __forceinline__ static char2 make(schar x, schar y) {return make_char2(x, y);}
        __host__ __device__ __forceinline__ static char2 make(const schar* v) {return make_char2(v[0], v[1]);}
        __host__ __device__ __forceinline__ static char2 make(char1 v, schar y=0) {return make_char2(v.x, y);} \
        __host__ __device__ __forceinline__ static reduct_elem_type reduce(const char2 v) {return reduct_elem_type(v.x)+v.y;}
    };
    template<> struct VecTraits<char3>
    {
        typedef schar elem_type;
        typedef int reduct_elem_type;
        enum {cn=3};
        __host__ __device__ __forceinline__ static char3 all(schar v) {return make_char3(v, v, v);}
        __host__ __device__ __forceinline__ static char3 make(schar x, schar y, schar z) {return make_char3(x, y, z);}
        __host__ __device__ __forceinline__ static char3 make(const schar* v) {return make_char3(v[0], v[1], v[2]);}
        __host__ __device__ __forceinline__ static char3 make(char2 v, schar z=0) {return make_char3(v.x, v.y, z);}
        __host__ __device__ __forceinline__ static char3 make(char1 v, schar y=0, schar z=0) {return make_char3(v.x, y, z);}
        __host__ __device__ __forceinline__ static reduct_elem_type reduce(const char3 v) {return reduct_elem_type(v.x)+v.y+v.z;}
    };
    template<> struct VecTraits<char4>
    {
        typedef schar elem_type;
        typedef int reduct_elem_type;
        enum {cn=4};
        __host__ __device__ __forceinline__ static char4 all(schar v) {return make_char4(v, v, v, v);}
        __host__ __device__ __forceinline__ static char4 make(schar x, schar y, schar z, schar w) {return make_char4(x, y, z, w);}
        __host__ __device__ __forceinline__ static char4 make(const schar* v) {return make_char4(v[0], v[1], v[2], v[3]);}
        __host__ __device__ __forceinline__ static char4 make(char3 v, schar w=0) {return make_char4(v.x, v.y, v.z, w);} \
        __host__ __device__ __forceinline__ static char4 make(char2 v, schar z=0, schar w=0) {return make_char4(v.x, v.y, z, w);} \
        __host__ __device__ __forceinline__ static char4 make(char1 v, schar y=0, schar z=0, schar w=0) {return make_char4(v.x, y, z, w);} \
        __host__ __device__ __forceinline__ static reduct_elem_type reduce(const char4 v) {return reduct_elem_type(v.x)+v.y+v.z+v.w;}
    };

}} // namespace lv::cudev
