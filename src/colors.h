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
#ifndef COLORS_H
#define COLORS_H

#include "math_linear.h" // vec_t

//BITROT: useless.
namespace colors {
	struct rgb_t : vec_t {
		rgb_t(float r, float g, float b) : vec_t(r/255,g/255,b/255) {}
		explicit rgb_t(const vec_t &v) : vec_t(v) {}
	};
	static const rgb_t 
		// http://en.wikipedia.org/wiki/Khaki_(color)
		khaki( 189, 183, 107), // <- dark variant.
		sepia(112, 66, 20),
		yellow(255, 255,   0),
		white( 255, 255, 255),
		almost(white*.999f);
}
#endif