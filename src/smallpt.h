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
#ifndef SMALLPT_H
#define SMALLPT_H

#include "specifics.h"

// interop header.
namespace smallpt {
	typedef float3 pixel_t;
	typedef float  pixel_cpnt_t;
}
#if 1
	#define PIXEL_TYPE			float3
	#define PIXEL_CPNT_TYPE		float
	#define PIXEL_GL_TYPE		GL_FLOAT
#else
	#define PIXEL_TYPE			uchar3
	#define PIXEL_CPNT_TYPE		uint8_t
	#define PIXEL_GL_TYPE		GL_UNSIGNED_BYTE
#endif

struct render_params_t {
	smallpt::pixel_t *framebuffer;
	float gamma;				// 0 to disable.
	unsigned regs_per_block;	// so we can decide to throw more threads at it.
	// toggles
	unsigned is_progressive;
	unsigned verbose;
	// 
	unsigned pass;				// pass #, for progressive refinement. pass == 0 will naturally signal a reset.
	unsigned spp;
	unsigned max_paths;
	unsigned num_spheres;
};

// CUDA interop.
extern "C" void *smallpt_make(unsigned w, unsigned h);
extern "C" void smallpt_destroy(void *smallpt);
extern "C" float smallpt_render(void *smallpt, const render_params_t *rp);
extern "C" void cuda_upload_camera(const vec_t cam[4]);
#ifdef __CUDACC__
	extern "C" void cuda_upload_scene(const size_t num_spheres, const sphere_t *spheres);
#else
	extern "C" void cuda_upload_scene(const size_t num_spheres, const cuda_sphere_t *spheres);
#endif

extern "C" void fatal(const char * const);

namespace sys {
	class fmt_t {
		enum { buffer_size = 256-sizeof(int) };
		char buffer[buffer_size];
		int len;
	public:
		explicit fmt_t(const char * __restrict const fmt, ...);
		const char *get() const { return buffer; }
		operator const char *() const { return get(); }
		size_t size() const { return len; }
	};
}

#endif