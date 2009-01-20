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
#ifndef RT_HIT_H
#define RT_HIT_H

// moved to this pseudo header for clarity's sake.

#include "specifics.h"
#include "math_linear.h"
#include "rt_ray.h"
#include "rt_sphere.h"


//WTF:
// there's no good reason for intersection functions to be members, but that
// nvcc gets upset if they're not; go figure.

struct hit_t {
	unsigned id;
	__device__ hit_t(unsigned sphere_id) : id(sphere_id) {}
	__device__ const sphere_t *operator ->() const { return scene.index(id); }

	// intersect and move ray origin (simplifies stuff).
	// faster predicated version.


	//
	// unrolled version
	//
	template<bool allow_mad, unsigned n>
		static __device__ void intersect_unrolled(const ray_t &ray, float &t, unsigned &id) {
			enum { i = n - 1 };
			intersect_unrolled<allow_mad, i>(ray, t, id);

			const sphere_t &sphere(*scene.index(i));
			const float eps = 1.f/(1<<14);
			const vec_t p(sphere.p - ray.o);
			// float b = DOT(p, ray.d), d = b*b - DOT(p, p) + sphere.radsqr;
			// float b = p.dot<false>(ray.d), d = math::muls(b, b) - p.dot<false>(p) + sphere.radsqr;
			float b, d;
			if (allow_mad) {
				b = p.dot<true>(ray.d);
				d = b*b - p.dot<true>(p) + sphere.radsqr;
			}
			else {
				b = p.dot<false>(ray.d);
				d = math::muls(b, b) - p.dot<false>(p) + sphere.radsqr;
			}
			bool_t bingo = d >= 0;
			d = math::sqrt(d);
			const float t1 = b-d < eps ? b+d : b-d;
			bingo = bingo && (t1 >= eps) && (t1 < t); // if you say so.
			if (bingo) t = t1;
			if (bingo) id = i;
		}
	#ifndef UNIX
		//WTF: need to re-specify linkage or else...
		template<> static __device__ void intersect_unrolled<false, 0>(const ray_t &ray, float &t, unsigned &id) {}
		template<> static __device__ void intersect_unrolled<true, 0>(const ray_t &ray, float &t, unsigned &id) {}
	#endif

	template<unsigned num_spheres>
	static __device__ hit_t intersect(ray_t &ray) {
		float t = 1e20f;
		unsigned id = ~0u;
		intersect_unrolled<true, num_spheres>(ray, t, id);
		ray.o = ray.advance(t);
		return hit_t(id);
	}

	//
	// loop version
	//
	static __device__ float pick_smallest_positive(float inf, float eps, float b, float d) {
		float t1 = b-d, t2 = b+d;
		if (t1 < eps) t1 = inf;
		if (t2 < eps) t2 = inf;
		return math::min(t1, t2);
	}
	static __device__ hit_t intersect(const unsigned &num_spheres, ray_t &ray) {
		const sphere_t * const spheres = scene.index(0);
		const float inf = CUDART_INF_F, eps = 1.f/(1<<14);
		float t = inf;
		unsigned id = ~0u;
		for (unsigned i=0; i<num_spheres; ++i) {
			const vec_t p(spheres[i].p - ray.o);
			float dot_pp = p.dot<1>(p), dot_pd = p.dot<1>(ray.d);
			float b = dot_pd, bsqr = math::mulm(b, b);
			float d = bsqr - dot_pp + spheres[i].radsqr;
			bool_t bingo = d >= 0;
			d = math::sqrt(d);
			float t1 = pick_smallest_positive(inf, eps, b, d);
			bingo = bingo && (t1 < t); // if you say so.
			if (bingo) t = t1;
			if (bingo) id = i;
		}
		if (id != ~0u) ray.o = ray.advance(t);
		return hit_t(id);
	}
};
#ifdef UNIX
	template<> void hit_t::intersect_unrolled<false, 0>(const ray_t &ray, float &t, unsigned &id) {}
	template<> void hit_t::intersect_unrolled<true, 0>(const ray_t &ray, float &t, unsigned &id) {}
#endif

#endif
