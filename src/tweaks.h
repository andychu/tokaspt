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
#ifndef TWEAKS_H
#define TWEAKS_H

//
// tweaks
enum {
	SS = 2,	SSsqr = SS*SS,	// super sampling (don't touch it for now).
	warp_size = 32,

	// larger = better, but it's incovenient.
	tile_size_x = 256, tile_size_y = tile_size_x, tile_area = tile_size_x*tile_size_y
};

enum { check_kernel_calls = 0 };

namespace scenes {
	enum {
		// constant mem allocation.
		// constant mem = 16k, 16k/64 = 256, give or take.
		max_num_spheres = 255
	};
}


// ...because we really only think in terms of tiles
namespace tweaks {
	template<typename T> static T round_up(T unit, T val) { return unit*(val/unit + (val % unit ? 1 : 0)); }
	template<typename T> static T make_res_x(T s) { return round_up(T(tile_size_x), s); }
	template<typename T> static T make_res_y(T s) { return round_up(T(tile_size_y), s); }
}

#endif
