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
#ifndef GL_CAMERA_H
#define GL_CAMERA_H

#include "specifics.h"
#include "math_linear.h"

// just to pull glFrustum in. oh my.
#include <GL/glew.h>
#include <GL/glut.h>

class scene_t;
namespace gl {
class camera_t;
	// a virtual screen, rays will be shot through from corner, up to corner+(dx*res.x, dy*res.y).
	// among other things, allows to easily generate ray (directions) the same way the rt side does.
	struct screen_sampler_t {
		vec_t corner, dx, dy;
		vec_t map(const point_t &point) const { return vec_t(corner + dx*float(point.x) + dy*float(point.y)); }
	};


	// let's keep everything extremely simple: a basis and an origin.
	// add a notion of up for the world, a fov and we have everything we need to:
	// a) tell the rt what to shoot b) match that for GL.
	class camera_t {
		friend class ::scene_t; // pff.
		static const vec_t world_ups[6];

		vec_t lft, up, fwd;
		vec_t eye;
	public: // direct access to fovy for the UI (would be a pain to do the Right Way[tm]).
		float fovy; // vertical fov in degree.
		enum world_up_t { UP_Y = 0, UP_Z, UP_X, UP_NY, UP_NZ, UP_NX, UP_LAST };
		world_up_t wu;

		// to make things more convenient for the UI, flip rendered image vertically.
		enum { flip_y = true };

		void set_dirty() {} // for simplicity sake, we'll detect that more brutally.
		void set_wu(world_up_t w)		{ set_dirty(); wu = w; }
		void set_eye(const vec_t &v)	{ set_dirty(); eye = v; }

		const vec_t &get_eye() const { return eye; }
		const vec_t &get_lft() const { return lft; }
		const vec_t &get_up() const { return up; }
		const vec_t &get_fwd() const { return fwd; }
		const vec_t &get_world_up() const { return world_ups[wu]; }
		float get_fovy() const { return fovy; }

		//adhoc crap
		void cycle_world_up() {
			wu = world_up_t((wu+1) % UP_LAST);
			look_at(eye + fwd);
		}
		// look at pos.
		// as we'll also use it to sanitize loaded cameras, make it a bit more robust.
		void look_at(const vec_t &pos) {
			fwd = (pos - eye).norm();
			if (math::abs(fwd.dot(world_ups[wu])) < .99f)
				lft = cross(world_ups[wu], fwd).norm();
			else {
				// gimbal lock, nudge.
				lft = vec_t(0, 0, 0);
				for (unsigned i=0; i<3; ++i)
					if (math::abs(world_ups[wu][i]) == 1) {
						lft[(i+1)%3] = 1;
						break;
					}
			}
			up  = cross(fwd, lft).norm();
			set_dirty();
		}

		// GL: set the projection.
		void set_frustum(const float aspect_ratio /* w/h */, const float t_near, const float t_far) const {
			const float t = math::tan(math::to_radian(fovy)/2);
			const float h = t_near*t;			// height of near plane / 2.
			const float w = h*aspect_ratio;	// width of near_place / 2.

			glFrustum(-w, w, -h, h, t_near, t_far);
		}

		// GL: make a view matrix, look into -Z.
		mat4_t make_view() const {
			const vec_t
				zaxis(-fwd),
				xaxis(-cross(world_ups[wu], zaxis).norm()),
				yaxis(cross(zaxis, -xaxis).norm());

			return mat4_t::from_axis_inv(xaxis, yaxis, zaxis, eye);
		}

		// RT: bake ray gen parameters.
		screen_sampler_t make_sampler(const point_t &res) const {
			const float aspect = float(res.x)/float(res.y);
			const float height = math::tan(math::to_radian(fovy)/2);
			const float width  = height*aspect;
			screen_sampler_t s;
			const float sdx = 2.f/res.x, sdy = 2.f/res.y;
			const float t_near = 1;
			const float flip = flip_y ? -1 : +1; // note: if we flip here, also flip the textured quad.
			s.corner = fwd*t_near - up*height*flip - lft*width;
			s.dx  = lft*width*sdx;
			s.dy  = up*height*sdy*flip;
			return s;
		}

		// reconstruct a proper camera.
		static camera_t make(const vec_t &pos, const vec_t &dir, const vec_t &up, const float fovy, const world_up_t wu) {
			camera_t c;
			c.lft = cross(world_ups[wu], dir).norm();
			c.up  = up;
			c.fwd = dir;
			c.eye = pos;
			c.fovy = fovy;
			c.wu = wu;
			return c;
		}
	};

}
#endif
