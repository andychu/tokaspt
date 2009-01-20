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
#ifndef MATH_LINEAR_H
#define MATH_LINEAR_H

#include "specifics.h"
#include "tweaks.h" // num_threads

#ifndef __CUDACC__
	#ifdef _MSC_VER
		#include <float.h> // _isnan, because there's no std::isnan.
	#endif
	#define _USE_MATH_DEFINES // M_PI & co.
#endif
#include <cmath>


#ifdef __CUDACC__
	// Host or Device
	#define HoD __device__
	// Host and Device
	#define HaD __device__ __host__
#else
	#include <limits> // aabb needs infinity.
	// and for that clueless msvc9...
	#define HoD FINLINE
	#define HaD FINLINE
#endif


// We need to be supah strict about exactly which function gets pulled; many env, much pain.
// Plus CUDA will force ie sin -> __sinf transformation if fast-math is on.
namespace math {
	#ifndef __CUDACC__
		// for a good laugh, see what msvc9 does about this. pathetic.
		static HoD float abs(float x) { return std::abs(x); }
		static HoD float square(float x) { return x*x; }
		static HoD float sqrt(float x) { return std::sqrt(x); }
		static HoD float rsqrt(float x) { return 1/math::sqrt(x); }
		static HoD float max(float x, float y) { return x > y ? x : y; }
		static HoD float min(float x, float y) { return x < y ? x : y; }

		static HoD float addm(float x, float y) { return x+y; }
		static HoD float subm(float x, float y) { return x-y; }
		static HoD float mulm(float x, float y) { return x*y; }
		static HoD float adds(float x, float y) { return x+y; }
		static HoD float subs(float x, float y) { return x-y; }
		static HoD float muls(float x, float y) { return x*y; }
		static HoD float div(float x, float y) { return x/y; }

		static HoD float frac(float val) { return val - int(val); }
		static HoD float tan(float val) { return std::tan(val); }
		static HoD float cot(float val) { return 1/std::tan(val); } // cotangent = 1/tan
		static HoD float to_radian(float val) { return val*M_PI/180; }
		// static HoD bool_t is_nan(float val) { return _isnan(val); } // msvc at it again.
	#else
		#if 0
			static __host__ __device__ float sin(float x) { return sinf(x); }
			static __host__ __device__ float cos(float x) { return cosf(x); }
		#else
			static  __device__ float sin(float x) { return __sinf(x); }
			static  __device__ float cos(float x) { return __cosf(x); }
		#endif

		// susceptible to m(ad) coalescing.
		static __device__ float addm(float x, float y) { return x+y; }
		static __device__ float subm(float x, float y) { return x-y; }
		static __device__ float mulm(float x, float y) { return x*y; }
		// not coalesced, let's say s(erialized).
		static __device__ float adds(float x, float y) { return __fadd_rn(x, y); }
		static __device__ float subs(float x, float y) { return __fadd_rn(x, -y); }
		static __device__ float muls(float x, float y) { return __fmul_rn(x, y); }
		// slow because of corner cases handling.
		static __device__ float div_slow(float x, float y) { return x/y; }
		// much faster but has issues with inf/large values and so on.
		static __device__ float div(float x, float y) { return __fdividef(x, y); }
		// (x*y) + z
		static __device__ float fma(float x, float y, float z) { return fmaf(x, y, z); }


		static __device__ float abs(float x) { return fabsf(x); }
		static __device__ float sqrt(float x) { return sqrtf(x); }
		static __device__ float rsqrt(float x) { return rsqrtf(x); }


		static __device__ float max(float x, float y) { return fmaxf(x,y); } //FIXME:
		static __device__ float min(float x, float y) { return fminf(x,y); }

		static __device__ float clamp(float x) { return __saturatef(x); }

		static __host__ float clamp_h(float x){ return x < 0 ? 0 : x > 1 ? 1 : x; }
	#endif
}



// at some point got tired of crummy template support (and typing).
#define V(op, lhs, rhs)		vec_t(op((lhs).x(), (rhs).x()), op((lhs).y(), (rhs).y()), op((lhs).z(), (rhs).z()))
#define Vs(op, lhs, rhs)	vec_t(op((lhs).x(), (rhs)), op((lhs).y(), (rhs)), op((lhs).z(), (rhs)))
#define H(op, rhs)			op(op((rhs).x(), (rhs).y()), (rhs).z())

// There's apparently some issues with member (as opposed to static) functions, dot was
// causing problems. Doesn't seem to trigger anymore. Phew.
//note: this is used for cuda { device + host }, and the native compiler; of course each path has its peculiarities.
//note: there's no FMA on current nvidia hardware, only MAC; they're labelled as FMA but there is an intermediary rounding
//      hence the fuss about allowing them or not (also, there's an additional undisclosed loss of precision). Yay.
struct vec_t {
	HoD vec_t() {}
	#ifdef __CUDACC__
		float3 m;
		HoD vec_t(float a, float b, float c) : m(make_float3(a,b,c)) {}
		// to ease proxification.
		HaD float x() const { return m.x; }
		HaD float y() const { return m.y; }
		HaD float z() const { return m.z; }
	#else
		// msvc9 fix. geez.
		float m[3];
		HoD vec_t(float a, float b, float c) { m[0]=a; m[1]=b; m[2]=c; }
		HoD explicit vec_t(const float * const p) { m[0]=p[0]; m[1]=p[1]; m[2]=p[2]; }
		HaD const float &x() const { return m[0]; }
		HaD const float &y() const { return m[1]; }
		HaD const float &z() const { return m[2]; }
		// well, since we now have an array, it's legit and convenient to
		const float &operator[](unsigned i) const { return m[i]; }
		float &operator[](unsigned i) { return m[i]; }
	#endif


	HoD vec_t operator+(const vec_t &rhs) const { return V(math::addm, *this, rhs); }
	HoD vec_t operator-(const vec_t &rhs) const { return V(math::subm, *this, rhs); }
	HoD vec_t operator*(const vec_t &rhs) const { return V(math::mulm, *this, rhs); }
	HoD vec_t operator*(const float rhs) const { return Vs(math::mulm, *this, rhs); }

	// shouldn't cause any trouble for this app.
	HoD vec_t operator/(const float rhs) const { return Vs(math::div, *this, rhs); }
	HoD vec_t operator-() const { return vec_t(-x(), -y(), -z()); }
	HoD float horizontal_max() const { return math::max(math::max(x(), y()), z()); }

	HoD vec_t norm() const { return *this * math::rsqrt(dot(*this)); }
	HoD vec_t reflect(const vec_t &n) const { return *this - n*dot(n)*2; }
	// mad allowed
	HoD float dot(const vec_t &rhs) const { return x()*rhs.x() + y()*rhs.y() + z()*rhs.z(); }
	template<bool allow_mad> HoD float dot(const vec_t &rhs) const {
		if (allow_mad)
			return x()*rhs.x() + y()*rhs.y() + z()*rhs.z();
		else
			return math::adds(math::adds(math::muls(x(), rhs.x()), math::muls(y(), rhs.y())), math::muls(z(), rhs.z()));
	}

	#ifndef __CUDACC__
		// nvcc doesn't get it and we don't need it for cuda anyway.
		friend HoD vec_t min(const vec_t &lhs, const vec_t &rhs) { return vec_t(math::min(lhs.x(), rhs.x()), math::min(lhs.y(), rhs.y()), math::min(lhs.z(), rhs.z())); }
		friend HoD vec_t max(const vec_t &lhs, const vec_t &rhs) { return vec_t(math::max(lhs.x(), rhs.x()), math::max(lhs.y(), rhs.y()), math::max(lhs.z(), rhs.z())); }
	#endif
};
static HoD vec_t cross(const vec_t &lhs, const vec_t &rhs) {
	return vec_t(lhs.y()*rhs.z() - lhs.z()*rhs.y(), lhs.z()*rhs.x() - lhs.x()*rhs.z(), lhs.x()*rhs.y() - lhs.y()*rhs.x());
}
// not quite what it claims to be, but matches what the original smallpt did.
static HoD vec_t make_basis(const vec_t &n) {
	bool_t which = math::abs(n.x()) > .1f;
	return cross(vec_t(which ? 0 : 1, which ? 1 : 0, 0), n).norm();
}
// only used on the host.
static HoD float dot(const vec_t &lhs, const vec_t &rhs) { return lhs.dot(rhs); }

static HoD vec_t reflect(const vec_t &d, const vec_t &n) { return d - n*d.dot(n)*2; }

#undef H
#undef V
#undef Vs

// nvcc is a glorified hack, macros to the rescue.
// #define DOT(a,b) dot(a, b) // better but triggers some bugs
// #define DOT(a,b) dot(a, b)

//#define REFLECT(d, n) d.do_reflect(n)
//#define REFLECT(d, n) reflect(d, n)

#define DOT(a,b)		a.dot<true>(b)
#define REFLECT(d, n)	d.reflect(n)

//
// float2 support
// i give up trying to template that shit.
//
// vector op vector (vertical)
#define OPv2(type, ctor, op, fun) static HoD type operator op(const type &lhs, const type &rhs) { return ctor(fun(lhs.x, rhs.x), fun(lhs.y, rhs.y)); }
// vector op scalar
#define OPs2(scalar, type, ctor, op, fun) static HoD type operator op(const type &lhs, const scalar &rhs) { return ctor(fun(lhs.x, rhs), fun(lhs.y, rhs)); }

OPv2(float2, make_float2, +, math::addm)
OPv2(float2, make_float2, -, math::subm)
OPv2(float2, make_float2, *, math::mulm)
OPv2(float2, make_float2, /, math::div)

OPs2(float, float2, make_float2, +, math::addm)
OPs2(float, float2, make_float2, -, math::subm)
OPs2(float, float2, make_float2, *, math::mulm)
OPs2(float, float2, make_float2, /, math::div)

#undef OPv2
#undef OPs2


// a thin proxy for a strided vector of 3 floats.
// actually nvcc seems to grok it, for a change.
template<block_size_t block_size>
struct strided_vec_t {
	#if __DEVICE_EMULATION__
		// emu bug. another.
		float *p;
		HoD strided_vec_t() :p(0) {}
	#else
		float * const p;
	#endif

	HoD strided_vec_t(float *q) : p(q) {}
	HoD float get(const unsigned idx) const { return p[block_size*idx]; }
	HoD void set(const unsigned idx, const float rhs) const { p[block_size*idx] = rhs; }

	HoD float x() const { return get(0); }
	HoD float y() const { return get(1); }
	HoD float z() const { return get(2); }

	HoD vec_t get() const { return vec_t(get(0), get(1), get(2)); }
	// HoD void set(const vec_t &rhs) { set(0, rhs.x); set(1, rhs.y); set(2, rhs.z); }
	HoD void set(const vec_t &rhs) { set(0, rhs.x()); set(1, rhs.y()); set(2, rhs.z()); }
	HoD void set(const float3 &rhs) { set(0, rhs.x); set(1, rhs.y); set(2, rhs.z); }


	// HoD operator vec_t() const { return vec_t(get(0), get(1), get(2)); }
	HoD operator vec_t() const { return get(); }
	HoD strided_vec_t &operator =(const vec_t &rhs) { set(rhs); return *this; }
};


#ifndef __CUDACC__
	//
	// vec4 / mat4
	//
	// minimal support & efforts, only used on the native side (for OpenGL etc...).
	//

	struct vec4_t {
		float m[4];
		vec4_t() {}
		vec4_t(float a, float b, float c, float d) { m[0]=a; m[1]=b; m[2]=c; m[3]=d; }
		explicit vec4_t(const vec_t &v, float d) { m[0]=v.x(); m[1]=v.y(); m[2]=v.z(); m[3]=d; }
		const float &operator[](unsigned i) const { return m[i]; }
		      float &operator[](unsigned i)       { return m[i]; }
		friend vec4_t operator +(const vec4_t &lhs, const vec4_t &rhs) { return vec4_t(lhs[0]+rhs[0], lhs[1]+rhs[1], lhs[2]+rhs[2], lhs[3]+rhs[3]); }
		friend vec4_t operator -(const vec4_t &lhs, const vec4_t &rhs) { return vec4_t(lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2], lhs[3]-rhs[3]); }
		friend vec4_t operator *(const vec4_t &lhs, const float scalar) { return vec4_t(lhs[0]*scalar, lhs[1]*scalar, lhs[2]*scalar, lhs[3]*scalar); }
		static vec_t to_vec3(const vec4_t &v) { return vec_t(v[0], v[1], v[2]); }
	};

	static float dot(const vec4_t &lhs, const vec4_t &rhs) {
		float r = 0;
		for (unsigned i=0; i<4; ++i)
			r += lhs[i]*rhs[i];
		return r;
	}



	// 4x4 matrix, columns vectors, column-major ordering.
	//	A -> B -> C transform = C*B*A
	//	A*v is v' as v transformed by A.
	struct mat4_t {
		vec4_t m[4];
		mat4_t() {}
		mat4_t(const vec4_t &v0, const vec4_t &v1, const vec4_t &v2, const vec4_t &v3) { m[0]=v0; m[1]=v1; m[2]=v2; m[3]=v3; }
		// wrong but it's only for throwing at OpenGL.
		const float *get() const { return static_cast<const float*>(m[0].m); }

		const vec4_t &operator[](unsigned i) const { return m[i]; }
		      vec4_t &operator[](unsigned i)       { return m[i]; }
		// vec4_t row(unsigned i) const { return vec4_t(m[0][i], m[1][i], m[2][i], m[3][i]); }

		friend mat4_t operator-(const mat4_t &lhs, const mat4_t &rhs) { return mat4_t(lhs[0]-rhs[0], lhs[1]-rhs[1], lhs[2]-rhs[2], lhs[3]-rhs[3]); }

		friend vec4_t operator*(const vec4_t &v, const mat4_t &mat) { return vec4_t(dot(v, mat[0]), dot(v, mat[1]), dot(v, mat[2]), dot(v, mat[3])); }
		friend vec4_t operator*(const mat4_t &mat, const vec4_t &v) { return mat[0]*v[0] + mat[1]*v[1] + mat[2]*v[2] + mat[3]*v[3]; }
		friend vec_t operator*(const mat4_t &mat, const vec_t &v) { return vec4_t::to_vec3(mat*vec4_t(v[0],v[1],v[2], 0)); }

		friend mat4_t operator*(const mat4_t &lhs, const mat4_t &rhs) {
			mat4_t r;
			for (unsigned i=0; i<4; ++i)
				for (unsigned j=0; j<4; ++j) {
					r[i][j] = 0;
					for (unsigned k=0; k<4; ++k)
						r[i][j] += lhs[k][j]*rhs[i][k];
				}
			return r;
		}

		static mat4_t from_axis(const vec_t &xaxis, const vec_t &yaxis, const vec_t &zaxis, const vec_t &t) {
			return mat4_t(vec4_t(xaxis, 0), vec4_t(yaxis, 0), vec4_t(zaxis, 0), vec4_t(t, 1));
		}
		static mat4_t from_axis_inv(const vec_t &xaxis, const vec_t &yaxis, const vec_t &zaxis, const vec_t &eye) {
			return mat4_t(
				vec4_t(xaxis.x(), yaxis.x(), zaxis.x(), 0),
				vec4_t(xaxis.y(), yaxis.y(), zaxis.y(), 0),
				vec4_t(xaxis.z(), yaxis.z(), zaxis.z(), 0),
				vec4_t(-dot(xaxis, eye), -dot(yaxis, eye), -dot(zaxis, eye) , 1) );
		}
	};


	// a simple 2D integral coordinate.
	struct point_t {
		int x, y;
		point_t(int xx, int yy) : x(xx), y(yy) {}
		int area() const { return x*y; }
		friend point_t operator-(const point_t &lhs, const point_t &rhs) { return point_t(lhs.x-rhs.x, lhs.y-rhs.y); }
		friend bool_t operator ==(const point_t &lhs, const point_t &rhs) { return lhs.x == rhs.x && lhs.y == rhs.y; }
		friend bool_t operator !=(const point_t &lhs, const point_t &rhs) { return !(lhs == rhs); }
	};

	// simple axis aligned bounding box.
	struct aabb_t {
		vec_t m[2]; // low, high.
		aabb_t() {}
		aabb_t(const vec_t &low, const vec_t &high) { m[0] = low; m[1] = high; }
		const vec_t &operator[](unsigned i) const { return m[i]; }
		vec_t extent() const { return m[1] - m[0]; }
		friend aabb_t compose(const aabb_t &lhs, const aabb_t &rhs) { return aabb_t(min(lhs[0], rhs[0]), max(lhs[1], rhs[1])); }
		static aabb_t infinite() {
			float inf = std::numeric_limits<float>::infinity();
			return aabb_t(vec_t(inf, inf, inf), -vec_t(inf, inf, inf));
		}
	};
#endif // !__CUDACC__

#endif
