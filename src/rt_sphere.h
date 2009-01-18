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
#ifndef RT_SPHERE_H
#define RT_SPHERE_H

#include "specifics.h"
#include "math_linear.h" // vec_t

enum refl_t { DIFF, SPEC, REFR };  // material types, used in radiance()

//
// to the cuda side, it's known as sphere_t.
// to the native, cuda_sphere_t.
//
#ifdef __CUDACC__
	struct ALIGN(16) sphere_t {
#else
	struct cuda_sphere_t {
#endif
		// 48 bytes
		vec_t p, e, c;			// position, emission, color
		float radsqr, max_c;	// squared radius, max color component.
		refl_t refl;			// reflection type (DIFFuse, SPECular, REFRactive)
		char pad[64-48];
};


#endif
