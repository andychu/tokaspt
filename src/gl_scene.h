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
#ifndef GL_SCENE_H
#define GL_SCENE_H

#include "specifics.h"
#include "math_linear.h"
#include "gl_camera.h"
#include "rt_sphere.h" // refl_t
#include "smallpt.h"

#include <vector>
#include <cassert>


// ===========================================================================
//
//						scene support
//
// ===========================================================================
class scene_t;

class sphere_t {
public:
	typedef uint32_t tag_t;
	sphere_t() {}
	sphere_t(refl_t refl, float rayon, const vec_t &position, const vec_t &color, const vec_t &emission)
		:
			pos(position), emi(emission), col(color),
			rad(rayon), type(refl), tag(-1)
		{}

	// transition...
	static sphere_t make_old_sphere(float rr, const vec_t &p, const vec_t &e, const vec_t &c, refl_t refl) {
		const float merde = 1; // 1.f/32;
		return
			// sphere_t(refl, math::sqrt(rr), p, c, e);
			sphere_t(refl, rr*merde, p*merde, c, e);

	}

	// serialize a sphere_t into cuda form.
	static void make_cuda_sphere(const sphere_t &s, cuda_sphere_t *cs) {
		cs->p = s.pos;
		cs->e = s.emi;
		cs->c = s.col;
		cs->radsqr = math::square(s.rad);
		cs->max_c = s.col.horizontal_max();
		cs->refl = refl_t(s.type);
	}

	vec_t pos, emi, col;
	float rad;
	int type; // really a refl_t
private:
	tag_t tag;
};

// holds the whole scene and provides ways to manipulate it.
//FIXME: it's rather bloated now, brush it up.
class scene_t {
	typedef std::vector<sphere_t> spheres_t;

	// buffer for cuda transfers.
	template<typename T>
	struct cuda_buffer_t {
		T *mem;
		size_t capacity;
		cuda_buffer_t() : mem(0), capacity(0) {}
		~cuda_buffer_t() { free(mem); }
		T *get(size_t num) {
			if (num > capacity) {
				capacity = capacity < 16 ? 16 : capacity;
				while (capacity < num) capacity *= 2;
				mem = static_cast<T*>(realloc(mem, sizeof(T)*capacity));
			}
			return mem;
		}
	};


	spheres_t spheres;
	sphere_t clipboard;
	// while we're at it, have a global bounding box (only maintained per frame).
	// seems weird to have a bounding box for spheres, no? :)
	aabb_t loose_bb;

	cuda_buffer_t<cuda_sphere_t> cuda_buffer;

	unsigned sel; // selection id.
	bool dirty;	// avoid undue xfers.


	void add(const sphere_t &s) {
		spheres.push_back(s);
		sel = size()-1;
		set_dirty();
	}
	aabb_t bounding_box() const {
		aabb_t bb(aabb_t::infinite());
		for (spheres_t::const_iterator it(spheres.begin()); it != spheres.end(); ++it) {
			float r = it->rad;
			bb = compose(bb, aabb_t(it->pos - vec_t(r, r, r), it->pos + vec_t(r, r, r)));
		}
		return bb;
	}
public:

	scene_t() { reset(); }

	unsigned size() const { return unsigned(spheres.size()); }

	const sphere_t &get(size_t i) const { return spheres[i]; }
	sphere_t get(size_t i) { return spheres[i]; }

	// return a cached max extent for the whole scene.
	float estimate_max_extent() const { return loose_bb.extent().horizontal_max(); }

	void set_dirty() { dirty = true; }

	//
	// selection
	//
	void deselect() { sel = -1; }
	unsigned get_selection_id() const { return sel; }
	bool_t is_valid_selection() const { return sel < spheres.size(); }
	const sphere_t &selection() const { assert(is_valid_selection() && "bad selection, no go"); return spheres[sel]; }
	sphere_t &selection() { assert(is_valid_selection() && "bad selection, no go"); return spheres[sel]; }

	void reset() {
		deselect();
		set_dirty();
		clipboard = sphere_t(DIFF, math::sqrt(1.f/16), vec_t(0,0,0), vec_t(1,1,0), vec_t(0,0,0));
		loose_bb = aabb_t::infinite();
	}
	void clear() { spheres.clear(); reset(); }

	// set it up with a initial scenery, also prime a view (optionally load it from a file).
	void init(gl::camera_t &cam, const char * const scene_filename);
	// update/upload scene to the cuda side if needed
	void upload_if(render_params_t &params);
	// shoot at every sphere to find nearest / farthest (if nearer, farther).
	void estimate_near_far(const vec_t &pos, float &t_near, float &t_far);
	// pick a sphere for selection.
	void picking(const vec_t &ray_o, const vec_t &ray_d);

	void cycle_selection(int dir) {
		sel += dir;
		if (sel >= size()) // wrap
			sel = dir > 0 ? 0 : size()-1;
	}

	//FIXME: 1.f/128; ok, clean that [censored]
	void addd(const vec_t &position, float radius = math::sqrt(1.f/128)) {
		// more useful if using the cliboard.
		sphere_t s = clipboard;
		s.pos = position;
		s.rad = radius;
		add(s);
	}

	// delete selection.
	void del() {
		if (is_valid_selection()) {
			spheres.erase(spheres.begin() + sel);
			set_dirty();
		}
	}

	// copy selection to clipboard.
	void copy() {
		if (is_valid_selection())
			clipboard = selection();
	}

	// paste clipboard
	void paste() {
		add(clipboard);
	}

	// paste clipboard @ position.
	void paste(const vec_t &position) {
		sphere_t s(clipboard);
		s.pos = position;
		add(s);
	}

	// ask for a filename, save.
	bool_t save(const gl::camera_t &cam);
	// ask for a filename, try to load, iff successful replace the current one.
	bool_t load(gl::camera_t &cam);
	// try to load a scene, iff successful replace the current one.
	bool_t load(gl::camera_t &cam, const char * const filename);
	// transitional.
	bool_t load_old(gl::camera_t &cam);

	// englobe the whole scene.
	// either with a sphere (!hollow), or a bunch of sphere all around (hollow).
	void mdl_englobe(const bool hollow, const float space, const float scale) {
		if (size()) {
			printf("englobing[%d] space %f scale %f\n", hollow, space, scale);
			aabb_t bb(bounding_box());
			vec_t extent(bb.extent()/2);
			vec_t center((bb[0]+bb[1])/2); // could be a NaN (-inf + inf), if the scene were to be empty (but it's not).
			if (hollow) {
				float r = extent.horizontal_max()*2;
				r *= scale; // to reduce apparent curvature, use a bigger factor.
				// spawn 2 spheres on boundaries(+offset) of each axis.
				for (unsigned i=0; i<3; ++i) {
					vec_t offset(0, 0, 0);
					offset[i] = extent[i] + r;
					addd(center+offset, r);
					addd(center-offset, r);
				}
			}
			else {
				// simply put a big sphere around.
				float r = extent.horizontal_max()*math::sqrt(2)*scale;
				addd(center, r);
			}
		}
	}

	// refit.
	// either refit the whole scene within max_dim (is_fitting), or scale it (!is_fitting).
	// adjust camera when done.
	void mdl_refit(gl::camera_t &cam, const bool is_fitting, const float max_dim) {
		if (size()) {
			aabb_t bb(bounding_box());
			vec_t extent(bb.extent());
			float r = extent.horizontal_max()/2;
			float scale = max_dim/r;
			if (!is_fitting) scale = 1/max_dim; // we're not fitting but scaling.
			for (spheres_t::iterator it(spheres.begin()); it != spheres.end(); ++it) {
				sphere_t &s(*it);
				s.pos = s.pos*scale;
				s.rad = s.rad*scale;
			}
			cam.set_eye(cam.get_eye()*scale);
			set_dirty();
		}
	}
};

#endif
