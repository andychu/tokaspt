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
#ifndef PREDICATED_H 
#define PREDICATED_H 


//
// ternary operator screw-up fixing.
//
#if 1
template<typename T> static __device__ void cond_set(const bool_t p, T &dst, const T &src) { if (p) dst = src; }
/* NEIN! gcc...
template<> static __device__ void cond_set<vec_t>(const bool_t p, vec_t &dst, const vec_t &src) {
	if (p) dst.x = src.x;
	if (p) dst.y = src.y;
	if (p) dst.z = src.z;
}
*/

// builds something from 2 alternatives (note: false part conditionaly assigned).
template<typename T> static __device__ T cond_make(const bool_t p, const T &yes, const T &no) {
	T res(yes);
	cond_set(!p, res, no);
	return res;
}
#else
// oula malheureux.
template<typename T> static __device__ void cond_set(const bool_t p, T &dst, const T &src) {
	if (p) dst = src;
}

// builds something from 2 alternatives (note: false part conditionaly assigned).
template<typename T> static __device__ T cond_make(const bool_t p, const T &yes, const T &no) {
	return p ? yes : no;
}
#endif

#endif
