/*
	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License version 2
	as published by the Free Software Foundation.

	This program is distributed in the hope that it will be useful,
	but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program; if not, write to the Free Software
	Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA


	Copyright (C) 2009  Thierry Berger-Perrin <tbptbp@gmail.com>, http://ompf.org
*/
#ifndef FIXES_H_
#define FIXES_H_

// compiler / platform workarounds.
#ifndef __CUDACC__
	#ifdef _MSC_VER
		// MSVC
		#ifndef _CRT_SECURE_NO_WARNINGS
			#define _CRT_SECURE_NO_WARNINGS
		#endif
		#ifndef _SCL_SECURE_NO_WARNINGS
			#define _SCL_SECURE_NO_WARNINGS
		#endif
		#ifndef _SECURE_SCL
			#define _SECURE_SCL 0
		#endif

		#define ALIGN(x)		__declspec(align(x))
		#define BREAKPOINT()	__asm { int 3 }
		#define FINLINE			__forceinline
		#define NOINLINE

		#pragma warning(disable : 4244) // 'return' : conversion from 'double' to 'float', possible loss of data
		#pragma warning(disable : 4305) // 'initializing' : truncation from 'double' to 'float'
		#pragma warning(disable : 4146) // unary minus operator applied to unsigned type, result still unsigned
		#pragma warning(disable : 4530) // C++ exception handler used, but unwind semantics are not enabled. Specify /EHsc

		// where would we be without Standard compliance, eh?
		#define snprintf _snprintf
	#else
		// GCC
		#define ALIGN(x)		__attribute__((aligned(x)))
		#define BREAKPOINT()	asm("int3")
		#define FINLINE			__attribute__((always_inline))
		#define NOINLINE		__attribute__((noinline))
	#endif
#else
	// NVCC

	// #include <cutil_inline.h>
	// #include <sm_13_double_functions.h> // ok, that's not enough to get __double_as_longlong & co.
	// #include <math_functions_dbl_ptx3.h>

	#define ALIGN(x) __align__(x)
	#define BREAKPOINT()

	#define FINLINE __attribute__((always_inline))
	// #define FINLINE inline
	#define NOINLINE __noinline__
#endif

#ifndef __CUDACC__
	#include <cstddef> // size_t
#endif

// so that silly msvc can track those.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cutil.h>
#include <cutil_inline.h>

#include <math_constants.h> // CUDART constants.

typedef unsigned char uchar_t;
typedef unsigned char uint8_t;
typedef signed char int8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

//typedef uint16_t bool_t;
typedef uint32_t bool_t;

typedef uint16_t block_size_t;

#endif /* FIXES_H_ */
