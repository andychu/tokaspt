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
#include "specifics.h"

#include "math_linear.h"
#include "gl_scene.h"
#include "gl_camera.h"

#include "rt_sphere.h" // refl_t
#include "smallpt.h"



#include "colors.h"

#ifdef _WIN32
	#define STRICT			1
	#define VC_EXTRALEAN
	//#define WIN32_LEAN_AND_MEAN
	//#define WIN32_EXTRA_LEAN
	#define NOMINMAX			// weee
	#define _WIN32_WINNT	0x0501		// XP, 0x0500 -> Windows 2000
	#include <windows.h>
#endif
#include "platform.h"

// ===========================================================================
//
//						scene support
//
// ===========================================================================


namespace scenes {
	namespace details {
		static const float inf = std::numeric_limits<float>::infinity(), eps = 1.f/(1<<14);

		static float pick_smallest_positive(float infinity, float epsilon, float b, float d) {
			float t1 = b-d, t2 = b+d;
			if (t1 < epsilon) t1 = infinity;
			if (t2 < epsilon) t2 = infinity;
			return math::min(t1, t2);
		}
	}

	// intersect one sphere vs ray, updates t and return true on hit.
	// cut & paste ftw.
	static bool intersect(const vec_t &ray_o, const vec_t &ray_d, const sphere_t &sphere, float &t) {
		const vec_t p(sphere.pos - ray_o);
		float dot_pp = p.dot<1>(p), dot_pd = p.dot<1>(ray_d);
		float b = dot_pd, bsqr = math::square(b);
		float d = bsqr - dot_pp + math::square(sphere.rad);
		bool_t bingo = d >= 0;
		d = math::sqrt(d);
		float t1 = details::pick_smallest_positive(details::inf, details::eps, b, d);
		bingo = bingo && (t1 < t); // if you say so.
		if (bingo) {
			t = t1;
			return true;
		}
		else
			return false;
	}

	//HACK:
	template<typename T> static T refit(const T &thing) { return thing*1; }
}

namespace scenes {
	namespace ballzor {
		#define LARGE_R		1e3f //F(1e3)			// used to refurbish that scene for floats.
		#define V(a,b,c) vec_t(float(a),float(b),float(c))
		#define BR 100
		#define S sphere_t::make_old_sphere

		static const float master_scene_scale_factor = 1.f/32;
		// const vec_t color1(.25f, .5f, .75f), color2(.75f, .25f, .5f);
		const vec_t color1(1, 0, 0), color2(0, 0, 1);
		static const sphere_t init_spheres[] = {
			// radius, position, emission, color, material
			S(99,	vec_t(  0, 190,  0),	colors::khaki,	vec_t(0, 0, 0),			DIFF),//Lite
			S(BR,	vec_t(  0,-BR,  0),		vec_t(0,0,0),	vec_t(.75f,.75f,.75f),	DIFF),//Botm
			S(16,	vec_t( 16,  8,  0),		vec_t(0,0,0),	colors::white,			SPEC),//Mirr
			S(12,	vec_t(-16,  6,  0),		vec_t(0,0,0),	color1,					DIFF),//Glas
			S( 4,	vec_t(  0,  4,-12),		vec_t(0,0,0),	colors::almost,			REFR),//Glas
			S( 8,	vec_t(  0,  8,-65),		colors::white*.3,	colors::khaki,		DIFF)//mir back
			//sphere_t( 8,	vec_t(  0,  8,-65),		colors::khaki*.3,	colors::khaki,			DIFF),//mir back
			//sphere_t( 8,	vec_t(  0,  8,-65),		vec_t(0,0,0),	colors::khaki,			DIFF),//mir back
			//sphere_t( 8,	vec_t(  0,  8,-65),	vec_t(0,0,0),	yellow,					SPEC),//mir back
			//sphere_t( 8,	vec_t( -8,  8,+32),	yellow*.125f,	almost,					SPEC),//mir back
		};
		#undef S
		#undef BR
		#undef V
		#undef LARGE_R
		enum { num_spheres = sizeof(init_spheres)/sizeof(sphere_t) };
	} // namespace ballzor
	namespace refitted {
#define V(x, y, z) vec_t((x), (y), (z))
#define Cam(pos, dir, up, fovy) static const gl::camera_t cam = gl::camera_t::make(pos, dir, up, fovy, gl::camera_t::UP_NY);
// #define S(pos, emi, col, radsqr, refl) sphere_t((pos), (emi), (col), (radsqr), refl_t(refl))
// sphere_t(refl_t refl, float rayon, const vec_t &position, const vec_t &color, const vec_t &emission)


		Cam(V(+0.000000000, +0.040899795, -0.184049070), V(+0.000000000, -0.044400614, +0.999013782), V(-0.000000000, -0.999013901, -0.044400617),  +40.000000000)

#define S(pos, emi, col, rad, refl) sphere_t(refl_t(refl), (rad), (pos), (col), (emi)),
static const sphere_t spheres[] = {
		S(V(+0.000000000, +0.777096093, +0.000000000), V(+0.741176486, +0.717647076, +0.419607848), V(+0.000000000, +0.000000000, +0.000000000),  +0.404907972, 0)
		S(V(+0.000000000, -0.408997953, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+0.750000000, +0.750000000, +0.750000000),  +0.408997953, 0)
		S(V(+0.065439671, +0.032719836, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +1.000000000, +1.000000000),  +0.065439671, 1)
		S(V(-0.065439671, +0.024539877, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +0.000000000, +0.000000000),  +0.049079753, 0)
		S(V(+0.000000000, +0.016359918, -0.049079753), V(+0.000000000, +0.000000000, +0.000000000), V(+0.999000013, +0.999000013, +0.999000013),  +0.016359918, 2)
		S(V(+0.000000000, +0.032719836, -0.265848666), V(+0.300000012, +0.300000012, +0.300000012), V(+0.741176486, +0.717647076, +0.419607848),  +0.032719836, 0)
		};
		#undef V
		#undef S
		#undef Cam

	}

#define Cam(pos, dir, up, fovy, wu) static const gl::camera_t cam = gl::camera_t::make(pos, dir, up, fovy, gl::camera_t::world_up_t(wu));
#define Begin static const sphere_t spheres[] = {
#define End   };
#define V(x, y, z) vec_t((x), (y), (z))
#define S(pos, emi, col, rad, refl) sphere_t(refl_t(refl), (rad), (pos), (col), (emi)),

	namespace test0 {
		// camera(pos, dir, up, fovy, world_up)
		// sphere(pos, emi, col, rad, refl)
		// version 2.1
		Cam(V(+0.000000000, +0.040899795, -0.184049070), V(+0.000000000, -0.044400614, +0.999013782), V(-0.000000000, -0.999013901, -0.044400617),  +40.000000000, 3)
		Begin
			S(V(+0.000000000, +0.777096093, +0.000000000), V(+0.741176486, +0.717647076, +0.419607848), V(+0.000000000, +0.000000000, +0.000000000),  +0.404907972, 0)
			S(V(+0.000000000, -0.408997953, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+0.750000000, +0.750000000, +0.750000000),  +0.408997953, 0)
			S(V(+0.065439671, +0.032719836, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +1.000000000, +1.000000000),  +0.065439671, 1)
			S(V(-0.065439671, +0.024539877, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +0.000000000, +0.000000000),  +0.049079753, 0)
			S(V(+0.000000000, +0.016359918, -0.049079753), V(+0.000000000, +0.000000000, +0.000000000), V(+0.999000013, +0.999000013, +0.999000013),  +0.016359918, 2)
		End

	}
	namespace hall_of_mirrors {
		// camera(pos, dir, up, fovy, world_up)
		// sphere(pos, emi, col, rad, refl)
		// version 2.1
		Cam(V(-0.010855001, +0.004500066, -0.146973431), V(+0.069367580, -0.336245656, +0.939216077), V(-0.024766607, -0.941774368, -0.335332334),  +80.000000000, 3)
		Begin
		S(V(+0.000000000, +0.091423072, +0.000000000), V(+0.741176486, +0.717647076, +0.419607848), V(+0.000000000, +0.000000000, +0.000000000),  +0.047636233, 0)
		S(V(+0.000000000, -0.048117407, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+0.750000000, +0.750000000, +0.750000000),  +0.048117407, 0)
		S(V(+0.007698785, +0.003849393, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +1.000000000, +1.000000000),  +0.007698785, 1)
		S(V(-0.007698785, +0.002887044, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +0.000000000, +0.000000000),  +0.005774089, 0)
		S(V(+0.000000000, +0.001924696, -0.005774089), V(+0.000000000, +0.000000000, +0.000000000), V(+0.999000013, +0.999000013, +0.999000013),  +0.001924696, 2)
		S(V(+0.000000000, +0.003849393, -0.031276315), V(+0.300000012, +0.300000012, +0.300000012), V(+0.741176486, +0.717647076, +0.419607848),  +0.003849393, 0)
		S(V(+0.989293873, +0.021412246, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +1.000000000, +1.000000000),  +0.941176474, 1)
		S(V(-0.989293873, +0.021412246, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +1.000000000, +1.000000000),  +0.941176474, 1)
		S(V(+0.000000000, +1.080235720, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +1.000000000, +0.000000000),  +0.941176474, 0)
		S(V(+0.000000000, -1.037411332, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +1.000000000, +0.000000000),  +0.941176474, 0)
		S(V(+0.000000000, +0.021412246, +0.989293873), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +1.000000000, +0.000000000),  +0.941176474, 0)
		End
	}

	namespace simple_base {
		// camera(pos, dir, up, fovy, world_up)
		// sphere(pos, emi, col, rad, refl)
		// version 2.1
		Cam(V(-0.094580606, +0.297563404, -1.147418737), V(+0.097414680, -0.179302499, +0.978959143), V(+0.017754423, +0.983793974, +0.178421319),  +61.333335876, 0)
		Begin
		S(V(+0.000000000, +5.937500000, +0.000000000), V(+0.741176486, +0.717647076, +0.419607848), V(+0.000000000, +0.000000000, +0.000000000),  +3.093750000, 0)
		S(V(+0.000000000, -3.125000000, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+0.750000000, +0.750000000, +0.750000000),  +3.125000000, 0)
		S(V(+0.500000000, +0.250000000, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +1.000000000, +1.000000000),  +0.500000000, 1)
		S(V(-0.500000000, +0.187500000, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000), V(+1.000000000, +0.000000000, +0.000000000),  +0.375000000, 0)
		S(V(+0.000000000, +0.125000000, -0.375000000), V(+0.000000000, +0.000000000, +0.000000000), V(+0.999000013, +0.999000013, +0.999000013),  +0.125000000, 2)
		End
	}

	namespace beason {
		Cam(V(+0.000000000, +0.040899795, -0.184049070), V(+0.000000000, -0.044400614, +0.999013782), V(-0.000000000, -0.999013901, -0.044400617),  +40.000000000, 3)

		static const sphere_t spheres[] = {
			sphere_t(DIFF,  1e5, V( 1e5+1,40.8,81.6), V(.75,.25,.25), V(0,0,0)),//Left
			sphere_t(DIFF,  1e5, V(-1e5+99,40.8,81.6),V(.25,.25,.75), V(0,0,0)),//Rght
			sphere_t(DIFF,  1e5, V(50,40.8, 1e5),     V(.75,.75,.75), V(0,0,0)),//Back
			sphere_t(DIFF,  1e5, V(50,40.8,-1e5+170), V(0,0,0),       V(0,0,0)),//Frnt
			sphere_t(DIFF,  1e5, V(50, 1e5, 81.6),    V(.75,.75,.75), V(0,0,0)),//Botm
			sphere_t(DIFF,  1e5, V(50,-1e5+81.6,81.6),V(.75,.75,.75), V(0,0,0)),//Top
			sphere_t(SPEC, 16.5, V(27,16.5,47),       V(1,1,1)*.999,  V(0,0,0)),//Mirr
			sphere_t(REFR, 16.5, V(73,16.5,78),       V(1,1,1)*.999,  V(0,0,0)),//Glas
			sphere_t(DIFF,  600, V(50,681.6-.27,81.6),V(0,0,0),       V(12,12,12)) //Lite
		};
	}

	namespace beason_tweaked {
		// recipe: huge reduced from 1e5 to 1e3, then scale everything by 1/32
		//FIXME: put the original camera.
		Cam(V(+0.000000000, +0.040899795, -0.184049070), V(+0.000000000, -0.044400614, +0.999013782), V(-0.000000000, -0.999013901, -0.044400617),  +40.000000000, 3)
		const float huge = 1e3;
		static const sphere_t spheres[] = {
			sphere_t(DIFF, huge, V( huge+1,40.8,81.6), V(.75,.25,.25), V(0,0,0)),//Left
			sphere_t(DIFF, huge, V(-huge+99,40.8,81.6),V(.25,.25,.75), V(0,0,0)),//Rght
			sphere_t(DIFF, huge, V(50,40.8, huge),     V(.75,.75,.75), V(0,0,0)),//Back
			sphere_t(DIFF, huge, V(50,40.8,-huge+170), V(0,0,0),       V(0,0,0)),//Frnt
			sphere_t(DIFF, huge, V(50, huge, 81.6),    V(.75,.75,.75), V(0,0,0)),//Botm
			sphere_t(DIFF, huge, V(50,-huge+81.6,81.6),V(.75,.75,.75), V(0,0,0)),//Top
			sphere_t(SPEC, 16.5, V(27,16.5,47),        V(1,1,1)*.999,  V(0,0,0)),//Mirr
			sphere_t(REFR, 16.5, V(73,16.5,78),        V(1,1,1)*.999,  V(0,0,0)),//Glas
			sphere_t(DIFF,  600, V(50,681.6-.27,81.6), V(0,0,0),       V(12,12,12)) //Lite
		};
	}


	namespace beason_tweaked_scaled {
		// camera(pos, dir, up, fovy, world_up)
		// sphere(pos, emi, col, rad, refl)
		// version 2.1
		Cam(V(+1.696788788, +0.999110520, +5.258557320), V(+0.009823957, -0.049931560, -0.998704374), V(-0.000491138, -0.998752654, +0.049929142),  +40.000000000, 3)
		Begin
		S(V(+31.281250000, +1.274999976, +2.549999952), V(+0.000000000, +0.000000000, +0.000000000), V(+0.750000000, +0.250000000, +0.250000000),  +31.250000000, 0)
		S(V(-28.156250000, +1.274999976, +2.549999952), V(+0.000000000, +0.000000000, +0.000000000), V(+0.250000000, +0.250000000, +0.750000000),  +31.250000000, 0)
		S(V(+1.562500000, +1.274999976, +31.250000000), V(+0.000000000, +0.000000000, +0.000000000), V(+0.750000000, +0.750000000, +0.750000000),  +31.250000000, 0)
		S(V(+1.562500000, +1.274999976, -25.937500000), V(+0.000000000, +0.000000000, +0.000000000), V(+0.000000000, +0.000000000, +0.000000000),  +31.250000000, 0)
		S(V(+1.562500000, +31.250000000, +2.549999952), V(+0.000000000, +0.000000000, +0.000000000), V(+0.750000000, +0.750000000, +0.750000000),  +31.250000000, 0)
		S(V(+1.562500000, -28.700000763, +2.549999952), V(+0.000000000, +0.000000000, +0.000000000), V(+0.750000000, +0.750000000, +0.750000000),  +31.250000000, 0)
		S(V(+0.843750000, +0.515625000, +1.468750000), V(+0.000000000, +0.000000000, +0.000000000), V(+0.999000013, +0.999000013, +0.999000013),  +0.515625000, 1)
		S(V(+2.281250000, +0.515625000, +2.437500000), V(+0.000000000, +0.000000000, +0.000000000), V(+0.999000013, +0.999000013, +0.999000013),  +0.515625000, 2)
		S(V(+1.562500000, +21.291563034, +2.549999952), V(+12.000000000, +12.000000000, +12.000000000), V(+0.000000000, +0.000000000, +0.000000000),  +18.750000000, 0)
		End
	}

#undef Cam
#undef Begin
#undef End
#undef S
#undef V
	// namespace init = ballzor;
	// namespace init = refitted;
	// namespace init = test0;
	// namespace init = hall_of_mirrors;
	// namespace init = beason;
	// namespace init = beason_tweaked;
	// namespace init = beason_tweaked_scaled;
	namespace init = simple_base;
	enum { num_spheres = sizeof(init::spheres)/sizeof(sphere_t) };
}
















// ===========================================================================
//
//						scene support
//
// ===========================================================================


void scene_t::init(gl::camera_t &cam, const char * const scene_filename) {
	reset();
	if (!(scene_filename && load(cam, scene_filename))) {
		printf("scene_t::init: priming with builtin, %d spheres.\n", scenes::num_spheres);
		for (size_t i=0; i<scenes::num_spheres; ++i)
			spheres.push_back(scenes::init::spheres[i]);

		cam = scenes::init::cam;
	}
	loose_bb = bounding_box();
	set_dirty();
}



void scene_t::upload_if(render_params_t &params) {
	if (dirty) {
		const size_t num = size();
		cuda_sphere_t *cuda_spheres = cuda_buffer.get(num);
		// cook a buffer of spheres in suitable format for cuda kernels.
		for (size_t i=0; i<num; ++i)
			sphere_t::make_cuda_sphere(spheres[i], cuda_spheres + i);
		cuda_upload_scene(num, cuda_spheres);
		// loosely maintain a global bounding box.
		loose_bb = bounding_box();
	}

	params.num_spheres = size();
	dirty = false;
}


void scene_t::estimate_near_far(const vec_t &pos, float &t_near, float &t_far) {
	for (spheres_t::const_iterator it(spheres.begin()); it != spheres.end(); ++it) {
		vec_t d((it->pos - pos).norm());
		float t = scenes::details::inf;
		if (scenes::intersect(pos, d, *it, t)) {
			t_near = std::min(t_near, t);
			t_far  = std::max(t_far , t + it->rad*2);
		}
	}
}

void scene_t::picking(const vec_t &ray_o, const vec_t &ray_d) {
	float t = scenes::details::inf;
	deselect();
	for (size_t i=0; i<spheres.size(); ++i)
		if (scenes::intersect(ray_o, ray_d, spheres[i], t))
			sel = unsigned(i);
}

bool_t scene_t::load(gl::camera_t &cam) {
	char filename[platform::magic_fs_path_len];
	// if (platform::dialog_load_save<true>("Load what?", filename))
	platform::dialog_load_save<true>("Load what?", filename);
	return load(cam, filename);
}

bool_t scene_t::load(gl::camera_t &cam, const char * const filename) {
	if (FILE *file = fopen(filename, "rb")) {
		spheres_t rspheres;
		gl::camera_t rcam;
		sphere_t s;
		int stage = 0, num = 0;
		int got = 0, integral = 0;
		enum { max_len = 255 };
		char buf[max_len+1];
		#define V(vec) &vec[0], &vec[1], &vec[2]
		// braindead.
		while(fgets(buf, max_len, file) == buf) {
			buf[max_len] = 0;
			if (buf[0] != 'C' && buf[0] != 'S') continue; // yay.
			if (stage == 0) {
				got = sscanf(buf, "Cam(V(%f,%f,%f), V(%f,%f,%f), V(%f,%f,%f), %f, %d)", V(rcam.eye),  V(rcam.fwd), V(rcam.up), &rcam.fovy, &integral);
				if (got == 11) {
					rcam.wu  = gl::camera_t::world_up_t(integral);
					// sanitize
					rcam.look_at(rcam.eye + rcam.fwd);
					++stage;
					continue;
				}
			}
			else {
				got = sscanf(buf, "S(V(%f,%f,%f), V(%f,%f,%f), V(%f,%f,%f), %f, %d)", V(s.pos), V(s.emi), V(s.col), &s.rad, &integral);
				if (got == 11) {
					s.type = refl_t(integral);
					rspheres.push_back(s);
					++num;
					continue;
				}
			}
			printf("scene_t::load: stage:%d got:%d, perplexed by [%s]\n", stage, got, buf);
			stage = -1;
			break;
		}
		#undef V
		fclose(file);
		if (stage != -1 && num) {
			printf("scene_t::load[%s]: read %d spheres, and everything's fine.\n", filename, num);
			cam = rcam;
			spheres.swap(rspheres);
			set_dirty();
			return true;
		}
	}
	printf("scene_t::load: no go.\n");
	return false;
}

// try to load a scene, iff successful replace the current one.
bool_t scene_t::load_old(gl::camera_t &cam) {
	char filename[platform::magic_fs_path_len];
	if (platform::dialog_load_save<true>("Old load?", filename))
		if (FILE *file = fopen(filename, "rb")) {
			spheres_t rspheres;
			gl::camera_t rcam;
			sphere_t s;
			unsigned stage = 0, num = 0;
			int got = 0, integral = 0;
			enum { max_len = 255 };
			char buf[max_len+1];
			bool old_version_fixup = true;
			#define V(vec) &vec[0], &vec[1], &vec[2]
			// crude & unforgiving parsing.
			while(fgets(buf, max_len, file) == buf) {
				buf[max_len] = 0;
				if (buf[0] == '#') continue; // 'comment' at start of line.
				if (buf[0] == 'V') {
					// for now, only 2 versions, old one stored radius squared.
					old_version_fixup = false;
					continue;
				}
				else if (stage == 0) {
					got = sscanf(buf, "Cam(V(%f,%f,%f), V(%f,%f,%f), V(%f,%f,%f), %f)", V(rcam.eye),  V(rcam.fwd), V(rcam.up), &rcam.fovy);
					if (got == 10) {
						rcam.wu  = gl::camera_t::UP_NY;
						// sanitize
						rcam.lft = cross(gl::camera_t::world_ups[rcam.wu], rcam.fwd).norm();
						++stage;
						continue;
					}
				}
				else {
					//FIXME: rad*rad transition
					got = sscanf(buf, "S(V(%f,%f,%f), V(%f,%f,%f), V(%f,%f,%f), %f, %d)", V(s.pos), V(s.emi), V(s.col), &s.rad, &integral);
					if (got == 11) {
						if (old_version_fixup) s.rad = math::sqrt(s.rad);
						s.type = refl_t(integral);
						rspheres.push_back(s);
						++num;
						continue;
					}
				}
				printf("scene_t::load: stage:%d got:%d, perplexed by [%s]\n", stage, got, buf);
				stage = -1;
				break;
			}
			#undef V
			fclose(file);
			if (stage != -1u && num) {
				printf("scene_t::load[%s]: read %d spheres, and everything's fine.\n", filename, num);
				if (old_version_fixup) 	printf("scene_t::load: but it was an old crummy version, please save again.");
				cam = rcam;
				spheres.swap(rspheres);
				set_dirty();
				return true;
			}
		}
	printf("scene_t::load: no go.\n");
	return false;
}


bool_t scene_t::save(const gl::camera_t &cam) {
	// By now you've guessed i'm trying to make it both something that can be parsed or compiled. Yeah, i know.
	// Also, human are supposed to be able to read it.
	//note: world_up's not exported.
	static const char
		header[] =
			"// camera(pos, dir, up, fovy, world_up)\n"
			"// sphere(pos, emi, col, rad, refl)\n";

	char filename[platform::magic_fs_path_len];
	if (platform::dialog_load_save<false>("Save to?", filename))
		if (FILE *file = fopen(filename, "wb")) {
			fprintf(file, header);
			// we need 9 decimal digits to ensure float -> string -> float gives us back what we put in.
			#define Vf		"V(%+.9f, %+.9f, %+.9f), "
			#define Vv(v)	v.x(), v.y(), v.z()
			static const char fmt[] = "S(" Vf Vf Vf " %+.9f, %d)\n";
			fprintf(file, "// version 2.1\n");
			fprintf(file, "Cam(" Vf Vf Vf " %+.9f, %d)\n", Vv(cam.get_eye()), Vv(cam.get_fwd()), Vv(cam.get_up()), cam.get_fovy(), cam.wu);
			fprintf(file, "Begin\n");
			for (spheres_t::const_iterator it(spheres.begin()); it != spheres.end(); ++it) {
				const sphere_t &s(*it);
				fprintf(file, fmt, Vv(s.pos), Vv(s.emi), Vv(s.col), s.rad, s.type);
			}
			#undef Vf
			#undef Vv
			fprintf(file, "End\n");
			fclose(file);
			printf("scene_t::save[%s]: done.\n", filename);
			return true;
		}
	return false;
}

