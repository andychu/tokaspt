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
#ifndef RT_RAY_H
#define RT_RAY_H

#include "specifics.h"
#include "math_linear.h" // vec_t

struct ray_t {
	vec_t o, d;
	__device__ ray_t() {}
	__device__ ray_t(const vec_t &pos, const vec_t &dir) : o(pos), d(dir) {}
	__device__ vec_t advance(float t) const { return o + d*t; }
};

#endif
