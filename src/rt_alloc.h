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
#ifndef RT_ALLOC_H
#define RT_ALLOC_H

#include "specifics.h"
#include "smallpt.h" // interop

#include <cuda_runtime_api.h>

//
// make those calls a bit safer to use (and traceable).
//

template<bool device_side, typename T> 
	void cuda_allocate(T *&p, size_t n) {
		if (device_side)
			cudaMalloc((void**) &p, sizeof(T)*n);
		else
			cudaMallocHost((void**) &p, sizeof(T)*n);

		if (p == 0) 
			fatal(sys::fmt_t("*** cuda_allocate: failed to allocate %.3fM on %s\n", sizeof(T)*n/(1024.*1024), device_side ? "device" : "host"));
		#ifndef NDEBUG
			printf("allocated %.3fM on %s\n", (sizeof(T)*n)/(1024.*1024), device_side ? "device" : "host");
		#endif
	}
template<typename T>
	void cuda_deallocate(T *&p) {
		cudaFree(p);
		p = 0;
	}

template<typename T>
	void cuda_memset(T *p, size_t n) { 
		cudaMemset(p, 0, sizeof(T)*n);
	}
#endif
