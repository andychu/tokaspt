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
// nvcc: --ptxas-options=-v  // –maxrregcount=10 --keep -arch sm_11 --ptxas-options=-v
// __CUDACC__

#include "specifics.h"
#include "tweaks.h"

#include "math_linear.h"
#include "predicated.h" // helpers

#include "rt_ray.h"
#include "rt_sphere.h"
#include "rt_prng.h"
#include "rt_alloc.h"

#include "smallpt.h" // interop.

#include <cstring> // std::memset

// sorry for the use of macros, but nvcc is getting on my nerves.
#define TWO_PI			(2*CUDART_PI_F)


// CUDA can't deal with arrays of structs.
// Then you can't __align__(__alignof(T)).
// Since we only care about floats and NVidia should fix their compiler we'll go for...
template<typename T, size_t num>
	struct aligned_storage_t {
		// char ALIGN(4) raw[sizeof(T)*num];
		char ALIGN(16) raw[sizeof(T)*num];
		#if 1
			__device__ T *index(const unsigned i) { return reinterpret_cast<T*>(raw) + i; }
			__device__ const T *index(const unsigned i) const { return reinterpret_cast<const T*>(raw) + i; }
		#else
			// no good :(
			static __device__ unsigned offset(const unsigned i) { return __umul24(sizeof(T), i); }
			__device__ T *index(const unsigned i) { return reinterpret_cast<T*>(raw + offset(i)); }
			__device__ const T *index(const unsigned i) const { return reinterpret_cast<const T*>(raw + offset(i)); }
		#endif
	};

#if 0
	#ifdef NO_CUDA
		template<typename T, typename U>  static T bit_cast(const U val) { union { U u; T t; } bits = { val }; return bits.t; }
	#else
		template<typename T, typename U>  static __device__ T bit_cast(const U val);
		template<> static __device__ uint32_t bit_cast<uint32_t, float>(const float val) { return __float_as_int(val); }
		template<> static __device__ float bit_cast<float, uint32_t>(const uint32_t val) { return __int_as_float(val); }
		template<> static __device__ uint64_t bit_cast<uint64_t, double>(const double val) { return 0; /*__double_as_longlong(val); */ }
		template<> static __device__ double bit_cast<double, uint64_t>(const uint64_t val) { return 0; /* __longlong_as_double(val);*/ }
	#endif
#endif






// ===========================================================================
//						constant mem.
// ===========================================================================
__constant__ aligned_storage_t<sphere_t, scenes::max_num_spheres> scene;
__constant__ vec_t cu_cam[4]; // u, v, fwd, pos

//
// intersections (made into a pseudo header for clarity).
//
#include "rt_hit.h"


// ===========================================================================
//					radiance computation.
// ===========================================================================

// i'm getting tired of tinkering with those
// and i can't turn them into functions (nvcc ftw).
#if 1
#define CASE_SPEC() \
	{ ray.d = REFLECT(ray.d, n); }
#define CASE_DIFF() \
	{ \
		const float r1  = TWO_PI*rrrr.y, r2 = rrrr.z; \
		const float r2s = math::sqrt(r2); \
		const vec_t u(make_basis(nl)), v(cross(nl, u)); \
		ray.d = (u*math::cos(r1)*r2s + v*math::sin(r1)*r2s + nl*math::sqrt(1-r2)).norm(); \
	}
#define CASE_REFR() \
	{ \
		const float nc = 1, nt = 1.5f; \
		const float nnt = cond_make(into, nc/nt, nt/nc); \
		const float ddn = cond_make(into, dot_dn, -dot_dn); \
		const float cos2t = 1-nnt*nnt*(1-ddn*ddn); \
		if (cos2t < 0) ray.d = REFLECT(ray.d, n); /* T.I.R. */ \
		else {  \
			const vec_t tdir((ray.d*nnt - nl*(ddn*nnt + math::sqrt(cos2t))).norm()); \
			const float \
				a = nt-nc, b = nt+nc, \
				R0 = (a*a) / (b*b), \
				c = 1 - (into ? -ddn : DOT(tdir, n)), \
				Re = R0+(1-R0)*c*c*c*c*c, \
				P = .25f + .5f*Re, \
				Tr = 1-Re,  \
				RP = math::div(Re, P), \
				TP = math::div(Tr, 1-P); \
			bool_t pick = rrrr.z < P; \
			fac = cond_make(pick, fac*RP, fac*TP); \
			ray.d = cond_make(pick, REFLECT(ray.d, n), tdir); \
		} \
	}
#else
#define CASE_SPEC() \
	{ ray.d = REFLECT(ray.d, n); }
#define CASE_DIFF() \
	{ \
		const float r1  = TWO_PI*rrrr.y(), r2 = rrrr.z(); \
		const float r2s = math::sqrt(r2); \
		const vec_t u(make_basis(nl)), v(cross(nl, u)); \
		ray.d = (u*math::cos(r1)*r2s + v*math::sin(r1)*r2s + nl*math::sqrt(1-r2)).norm(); \
	}
#define CASE_REFR() \
	{ \
		const float nc = 1, nt = 1.5f; \
		const float nnt = cond_make(into, nc/nt, nt/nc); \
		const float ddn = cond_make(into, dot_dn, -dot_dn); \
		const float cos2t = 1-nnt*nnt*(1-ddn*ddn); \
		if (cos2t < 0) ray.d = REFLECT(ray.d, n); /* T.I.R. */ \
		else {  \
			const vec_t tdir((ray.d*nnt - nl*(ddn*nnt + math::sqrt(cos2t))).norm()); \
			const float \
				a = nt-nc, b = nt+nc, \
				R0 = (a*a) / (b*b), \
				c = 1 - (into ? -ddn : DOT(tdir, n)), \
				Re = R0+(1-R0)*c*c*c*c*c, \
				P = .25f + .5f*Re, \
				Tr = 1-Re,  \
				RP = math::div(Re, P), \
				TP = math::div(Tr, 1-P); \
			bool_t pick = rrrr.z() < P; \
			fac = cond_make(pick, fac*RP, fac*TP); \
			ray.d = cond_make(pick, REFLECT(ray.d, n), tdir); \
		} \
	}
#endif


// for this version, max_paths & scene size are known at compilation time.
template<unsigned max_paths, unsigned num_spheres, typename T, typename U>
static __device__ void radiance(const ray_t primary, T &rng, U &rad) {
	ray_t ray(primary);	// ray being traced around.
	vec_t fac(1, 1, 1);	// f being carried over.
	//rad: accumulated radiance.
	unsigned depth = 0 /* ray tree depth */;

	while (1) {
		const hit_t hit = hit_t::intersect<num_spheres>(ray);
		if (hit.id >= num_spheres) break;
		// pull 3 randoms now; unintuitive: they may not all be used, but then that makes our rng better and nvcc does a much better reg alloc
		// ah, but... now that it's all in global memory... hmm...
		const float3 rrrr(rng.gen3());

		const bool_t pred_d5 = ++depth > 5;
		const bool_t pred_stop = (pred_d5 && rrrr.x >= hit->max_c) || depth > max_paths;
		// update
		rad = fac*hit->e + rad;
		fac = fac*cond_make(!pred_d5, hit->c, hit->c / hit->max_c); // RRed for the 3rd case anyway.
		if (pred_stop) break;

		const vec_t n((ray.o - hit->p).norm());
		const float dot_dn = DOT(ray.d, n);
		const bool_t into = dot_dn < 0; // Ray from outside going in?
		const vec_t nl(cond_make(into, n, -n));

		switch(hit->refl) {
			case DIFF:	CASE_DIFF() break;
			case REFR:	CASE_REFR() break;
			default:	CASE_SPEC()
		}
	}
}

template<typename T, typename U>
static __device__ void radiance(const unsigned &num_spheres, const unsigned &num_paths, const ray_t primary, T &rng, U &rad) {
	ray_t ray(primary);	// ray being traced around.
	vec_t fac(1, 1, 1);	// f being carried over.
	//rad: accumulated radiance.
	unsigned depth = 0 /* ray tree depth */;

	while (1) {
		const hit_t hit = hit_t::intersect(num_spheres, ray);
		if (hit.id >= num_spheres) break;
		// pull 3 randoms now; unintuitive: they may not all be used, but then that makes our rng better and nvcc does a much better reg alloc
		// ah, but... now that it's all in global memory... hmm...
		const float3 rrrr(rng.gen3());

		const bool_t pred_d5 = ++depth > 5;
		const bool_t pred_stop = (pred_d5 && rrrr.x >= hit->max_c) || depth > num_paths;
		// update
		rad = fac*hit->e + rad;
		fac = fac*cond_make(!pred_d5, hit->c, hit->c / hit->max_c); // RRed for the 3rd case anyway.
		if (pred_stop) break;

		const vec_t n((ray.o - hit->p).norm());
		const float dot_dn = DOT(ray.d, n);
		const bool_t into = dot_dn < 0; // Ray from outside going in?
		const vec_t nl(cond_make(into, n, -n));

		switch(hit->refl) {
			case DIFF:	CASE_DIFF() break;
			case REFR:	CASE_REFR() break;
			default:	CASE_SPEC()
		}
	}
}

// ===========================================================================
//					ray gen, bookeeping.
// ===========================================================================
static __device__ float distribute(const float r /* [0, 2] */ ) {
	if (0) return cond_make(r < 1, math::sqrt(r)-1, 1-math::sqrt(2-r));
	else {
		const bool_t less_than_1 = r < 1;
		const float root = math::sqrt(cond_make(less_than_1, r, 2-r));
		const float eval = cond_make(less_than_1, root-1, 1-root);
		return eval;
	}
}

template<typename T>
static __device__ float2 distribute(T &rng_state) {
	const float2
		rand(rng_state.gen2()),
		twice(make_float2(rand.x*2, rand.y*2));
	return make_float2(distribute(twice.x), distribute(twice.y));
}


// reserve num_threads*(3 [+1])*4 bytes in shared mem:
// . 3 floats for a vector (radiance)
// . 1 pointer (optionnal) to access randoms
// with that last one (pointer), we get down to 24 registers
// and still have enough shared mem for 67% occupancy; sounds like the Right Thing[tm].
// untested yet tho.
template<block_size_t block_size>
struct ALIGN(16) shared_data_t {
	float floats[block_size*3];
	__device__ strided_vec_t<block_size> strided_vec(const unsigned x) { return strided_vec_t<block_size>(floats + x); }

	#ifdef RNG_BAKED_USES_SHARED
		// float *ptr[block_size];
		const float *ptr[block_size];
	#endif
};


// render/accumulate an anti-aliased tile of 1 sample.
template<block_size_t block_size, typename T> static __global__
	void smallpt_kernel(
		const unsigned num_spheres, const unsigned num_paths,	// seems to be the only way not to pay registers
		const unsigned should_clear_tile,						// for them.
		const float2 rcp_res,	/* 1/final image resolution */
		const uint2 tile,		/* tile corner, global coords */
		T *stuff,				/* either rng states or baked randoms */
		vec_t *tile_out			/* tile output */
	)
{
	__shared__ shared_data_t<block_size> shared_data;

	const uint32_t global_id = __umul24(block_size, blockIdx.x) + threadIdx.x;
	const uint32_t pidx = global_id / (SS*SS); // LSB = subpixel position.
	const uint32_t px = pidx % tile_size_x, py = pidx / tile_size_x; // coords within tile

	// rng setup.
	#ifdef RNG_BAKED_USES_SHARED
		// need to store that pointer in shared mem first.
		shared_data.ptr[threadIdx.x] = stuff + global_id;
		rng::proxy_t<block_size> rng_proxy(shared_data.ptr[threadIdx.x]);
	#else
		rng::proxy_t<block_size> rng_proxy(stuff + global_id);
	#endif

	// generate a primary
	const float2
		half(make_float2(.5f, .5f)),
		subrand(distribute(rng_proxy)),
		tilpix(make_float2(px + tile.x, py + tile.y)),							// pixel in global coords.
		subpix(make_float2(threadIdx.x & 1 ? +1 : 0, threadIdx.x & 2 ? +1 : 0)),// subpixel coords.
		pos((half + subpix + subrand)*half + tilpix);
	const vec_t d(cu_cam[0]*pos.x + cu_cam[1]*pos.y + cu_cam[2]);

	// radiance in shared mem is a cheap way to save quite some registers ;)
	// and quickly reduce subpixels.
	strided_vec_t<block_size> srad(shared_data.strided_vec(threadIdx.x));
	srad = vec_t(0, 0, 0);
	ray_t ray(cu_cam[3], d.norm());
	radiance(num_spheres, num_paths, ray, rng_proxy, srad);
	// rigged :)
	// radiance<smallpt::max_paths, smallpt::scene_num_spheres>(ray, rng_proxy, srad);

	// and now gather subpixels.
	__syncthreads();
    #if 0
        // coalesced writes, bank conflicts.
        // so far we've produced a block worth of super sampled samples with a base
        // tile offset of __umul24(block_size, blockIdx.x) / (SS*SS) in shared memory (strided).
        enum { num_cpnt = 3, num_subcpnt = num_cpnt*block_size/(SS*SS) };
        // to coalesce writes (once subsamples are accumulated), the idea is to remap
        // the bottom num_subcpnt threads into contiguous sample components.
        // trouble is, while we'll get coalesced writes, we'll also 
        // generate a metric ton of bank conflicts.
        if (threadIdx.x < num_subcpnt) {
            const float ss = 1.f/(SS*SS); // exact.
            const uint32_t idx_base = __umul24(block_size, blockIdx.x) / (SS*SS);
            const uint32_t major = threadIdx.x / num_cpnt;
            const uint32_t minor = threadIdx.x % num_cpnt;
            const uint32_t shared_offset = major*(SS*SS) + minor*block_size;

            float sum = 0;
            // nvcc has troubles with sum += shared_data.strided_vec(major*(SS*SS) + i).get(minor)*ss;
            // and wants another register for that. kludge.
            float * const p = reinterpret_cast<float*>(tile_out + idx_base) + threadIdx.x;
            if (!should_clear_tile) sum = *p;
            for (unsigned i=0; i<(SS*SS); ++i)
                // sum += shared_data.strided_vec(major*(SS*SS) + i).get(minor)*ss;
                sum += shared_data.floats[shared_offset + i]*ss;
            *p = sum;
        }
    #else
        // uncoalesced writes, no bank conflicts.
	    if ((threadIdx.x & ((SS*SS)-1)) == 0) {
		    const uint32_t idx = pidx; // already computed, __umul24(tile_size_x, py) + px;
		    const float ss = 1.f/(SS*SS); // exact.
		    vec_t sum(
			    vec_t(shared_data.strided_vec(threadIdx.x + 0))*ss +
			    vec_t(shared_data.strided_vec(threadIdx.x + 1))*ss +
			    vec_t(shared_data.strided_vec(threadIdx.x + 2))*ss +
			    vec_t(shared_data.strided_vec(threadIdx.x + 3))*ss );
		    // should we initialize that tile?
		    if (!should_clear_tile) sum = sum + tile_out[idx];
		    tile_out[idx] = sum;
	    }
    #endif
}
// ===========================================================================
//					helper kernels.
// ===========================================================================

#ifndef RNG_IS_BAKED
	//FIXME: horrible seeding, need to get back to previous performance level ASAP.
	static __global__ void init_online_rng(rng::state_t *rng_states) {
		const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;
		rng::seed(rng_states[tid], 0, tid);
	}
#endif


static __device__ float cumulative_moving_average(const float i, const float Ai, const float Xi) {
	return Ai + (Xi - Ai)/(i+1);
}

static __device__ float gamma_scale_round(float post_scale, float x){
	return __powf(math::clamp(x), 1/2.2f)*post_scale +.5f;
}

// compose a tile into the destination; try to coalesce read/write as much as possible (by going component wise).
template<bool cummulative, bool to_rgb8, typename T, typename U>
static __global__
	void compose_tile_linear(float iter, float rcp_num_samples, unsigned num_chunks, unsigned res_x3, unsigned offset3, const T *tile, U *pix) {
		unsigned stride = gridDim.x*blockDim.x;
		unsigned tid = blockDim.x*blockIdx.x + threadIdx.x;
		unsigned tile_width = 3*tile_size_x;
		unsigned i = num_chunks;
		do {
			uint32_t idx = offset3 + res_x3*(tid / tile_width) + (tid % tile_width);
			float val = tile[tid]*rcp_num_samples;
			if (cummulative) val = cumulative_moving_average(iter, pix[idx], val);
			if (to_rgb8) val = gamma_scale_round(255.f, val);
			pix[idx] = val;
			tid += stride;
		} while (--i);
	}

// in & out have the same dimensions.
static __global__ void postproc_linear(float gamma, unsigned num_chunks, const float *in, float *out) {
	uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;
	uint32_t stride = gridDim.x*blockDim.x;
	unsigned i = num_chunks;
	do {
		float val = in[tid];
		val = __powf(math::clamp(val), 1/gamma);
		out[tid] = val;
		tid += stride;
	} while (--i);
}


// ===========================================================================
//					host side
// ===========================================================================

static void progress_bar(const size_t num, const size_t total) {
	char progress[] = "|                                                                |\r";
	std::memset(progress + 1, '>', (num > total ? total : num)*64/total);
	printf(progress);
	fflush(stdout);
}

struct launcher_t {
	enum { num_streams = 1 };

	cudaStream_t streams[num_streams];
	// memory for the main rendering part, device side
	vec_t *d_tiles[num_streams];		// where tile samples are accumulated.
	rng::state_t *d_rng[num_streams];	// RNG states for each stream.
	rng::data_t *d_data[num_streams];	// and if that RNG bakes randoms, where they are stored.

	// we need to hold a full image on the device either:
	// . to accumulate for progressive refinment.
	// . to hold it before blitting to host when not doing prog ref.
	// the final (possibly post processed - ie gamma -) picture comes from the outside.
	typedef smallpt::pixel_t pixel_t;
	// typedef smallpt::pixel_cpnt_t pixel_cpnt_t;
	pixel_t *d_acc;

	const unsigned res_x, res_y, num_pixels, num_threads_per_tile /* a thread per supersampled pixel */;
	unsigned prev_max_paths; // because we need to reallocate if can be dynamically set.
	unsigned timer;

	launcher_t(unsigned w, unsigned h)
		:
			res_x(tweaks::make_res_x(w)), res_y(tweaks::make_res_x(h)),
			num_pixels(res_y*res_x), num_threads_per_tile(tile_area*SS*SS),
			prev_max_paths(0),
			timer(0)
	{
		allocate();
	}

	~launcher_t() { deallocate(); }

	void allocate() {
		// we'll produce a tile of pixels on each pass, with a thread per sub pixel,
		// we need to store that many rng states.
		for (unsigned i=0; i<num_streams; ++i) cudaStreamCreate(streams + i);
		// large allocations first.
		d_acc =  0;
		cuda_allocate<true>(d_acc, num_pixels);

		for (unsigned i=0; i<num_streams; ++i) cuda_allocate<true>(d_tiles[i], tile_area);

		// initialize rng states.
		#ifdef RNG_IS_BAKED
			// since max_paths may change, we'll delay d_data allocation.
			for (unsigned i=0; i<num_streams; ++i) d_data[i] = 0;
			for (unsigned i=0; i<num_streams; ++i) rng::init_baked_rng(i, d_rng[i]); // allocates d_rng
		#else
			// simply prime/seed states.
			//FIXME: should be primed differently among streams.
			for (unsigned i=0; i<num_streams; ++i) cuda_allocate<true>(d_rng[i], num_threads_per_tile);
			for (unsigned i=0; i<num_streams; ++i) init_online_rng<<<num_threads_per_tile/256, 256>>>(d_rng[i]);
			if (check_kernel_calls) cutilCheckMsg("'init_online_rng' kernel failure\n");
		#endif

		cutCreateTimer(&timer);
	}

	void deallocate() {
		#ifndef RNG_IS_BAKED
			for (unsigned i=0; i<num_streams; ++i) cuda_deallocate(d_rng[i]);
		#else
			for (unsigned i=0; i<num_streams; ++i) cuda_deallocate(d_data[i]);
		#endif
		for (unsigned i=0; i<num_streams; ++i) cuda_deallocate(d_tiles[i]);

		if (d_acc) cuda_deallocate(d_acc);

		for (unsigned i=0; i<num_streams; ++i) cudaStreamDestroy(streams[i]);
		cutDeleteTimer(timer);
	}

	float banzai(const render_params_t * const params) {
		// cuda occupancy calculator tells me: 64 threads for 8192regs, 128 for 16384 for the main kernel.
		return params->regs_per_block > 8192 ? banzai<128>(params) :  banzai<64>(params);
	}

	template<unsigned num_rendering_threads>
	float banzai(const render_params_t * const params) {
		enum { num_components = 3 };
		const unsigned
			num_samples = params->spp/(SS*SS) ? params->spp/(SS*SS) : 1, // # of supersampled samples.
			num_tiles_x = unsigned(res_x/tile_size_x), num_tiles_y = unsigned(res_y/tile_size_y),
			num_tiles = num_tiles_x*num_tiles_y;

		#ifdef RNG_IS_BAKED
			// how many randoms do we need per sample pass:
			// for the whole tile... 2 to generate a ray, 3 per path.
			//FIXME: something's fishy here (check for max_path = 1).
			const unsigned sanitized_max_paths = params->max_paths > 2 ? params->max_paths : 2;
			const unsigned want = tile_area*(2 + 3*sanitized_max_paths), data_size = rng::compute_data_size(want);
			if (prev_max_paths < sanitized_max_paths) {
				printf("baked rng: max_paths %d, want %d -> %.3fM, %d threads, %d chunks.\n",
					sanitized_max_paths, want, data_size/(1024.*1024), rng::param_n, rng::compute_num_chunks(want));
				for (unsigned i=0; i<num_streams; ++i) if (d_data[i]) cuda_deallocate(d_data[i]);
				for (unsigned i=0; i<num_streams; ++i) cuda_allocate<true>(d_data[i], data_size/sizeof(rng::data_t));
				prev_max_paths = sanitized_max_paths;
			}
		#endif

		if (params->verbose)
			printf("fasten your seatbelt, going for res(%d,%d) tiles(%d,%d)x(%d,%d)x%d....\n",
					res_x, res_y, num_tiles_x, num_tiles_y, SS, SS, num_samples);

		// when in progressive mode, we shoudln't really need to do that
		// but if the acc buffer contains some 'toxic' values for some reason, it may not work.
		if (params->pass == 0)
			cuda_memset(d_acc, num_pixels);


		cutResetTimer(timer);
		cutStartTimer(timer);

		const float2 rcp_res(make_float2(1.f/res_x, 1.f/res_y));
		const float rcp_num_samples(1.f/num_samples);
		// for each tile
		for (unsigned tile_num=0; tile_num<tweaks::round_up(unsigned(num_streams), num_tiles); tile_num += num_streams) {
			uint2 tiles[num_streams];		// tile top left corner coords.
			unsigned offsets[num_streams];	// offset of that tile into the whole picture.
			for (unsigned i=0; i<num_streams; ++i) {
				const unsigned tx = (tile_num+i) % num_tiles_x,  ty = (tile_num+i) / num_tiles_x;
				tiles[i] = make_uint2(tile_size_x*tx, tile_size_y*ty);
				offsets[i] = num_components*(res_x*tile_size_y*ty + tile_size_x*tx); // we'll go component wise
			}
			if (params->verbose) progress_bar(tile_num+num_streams, num_tiles);

			// for each sample: [generate some randoms], render an antialiased tile.
			const unsigned remaining_tiles = tile_num+num_streams > num_tiles ? num_tiles-tile_num : num_streams;
			for (unsigned s=0; s<num_samples; ++s)
				for (unsigned i=0; i<remaining_tiles; ++i) {
					#ifdef RNG_IS_BAKED
						rng::bake_some(streams[i], want, d_rng[i], d_data[i]);
					#endif
					smallpt_kernel<num_rendering_threads><<<num_threads_per_tile/num_rendering_threads, num_rendering_threads, 0, streams[i]>>>(
						params->num_spheres, params->max_paths, s == 0,
						rcp_res, tiles[i], d_data[i], d_tiles[i]);
				}

			// and then compose that tile.
			enum { grid_size = 64, block_size = 256, stride = grid_size*block_size };
			// how to remap a block of thread worth of tile pixel into the destination.
			const unsigned res_x3 = res_x*num_components, num_chunks = tile_area*num_components / stride;
			// printf("grid %d block %d total %d\n", grid_size, block_size, stride);
			if (params->is_progressive)
				for (unsigned i=0; i<remaining_tiles; ++i)
					compose_tile_linear<1, sizeof(d_acc[0].x) == 1><<<grid_size, block_size, 0, streams[i]>>>(
						float(params->pass), rcp_num_samples, num_chunks, res_x3, offsets[i], (const float*) d_tiles[i], &d_acc[0].x);
			else
				for (unsigned i=0; i<remaining_tiles; ++i)
					compose_tile_linear<0, sizeof(d_acc[0].x) == 1><<<grid_size, block_size, 0, streams[i]>>>(
						float(params->pass), rcp_num_samples, num_chunks, res_x3, offsets[i], (const float*) d_tiles[i], &d_acc[0].x);
		}
		cudaThreadSynchronize(); // not needed, it's just for better reporting.
		cutilCheckMsg("something went bad.\n");

		cutStopTimer(timer);
		float dt = cutGetTimerValue(timer);


		// at this point we either have an accumulated non post processed image
		// or the final post processed image  on device.
		if (params->is_progressive && params->gamma) {
			// post process
			enum { block_size = 192, grid_size = 128, num_threads = block_size*grid_size };
			const unsigned num_chunks = num_pixels*num_components / num_threads;
			// printf("postproc_linear... (%d,%d) = %d threads, %d chunks (gamma %f)\n", grid_size, block_size, num_threads, num_chunks, params->gamma);
			postproc_linear<<<grid_size, block_size>>>(params->gamma, num_chunks, (float*) &d_acc[0].x, (float*) params->framebuffer); // framebuffer type...
			cutilCheckMsg("postproc_linear, no good.\n");
		}
		else
			// final blit
			cudaMemcpy(params->framebuffer, d_acc, sizeof(pixel_t)*num_pixels, cudaMemcpyDeviceToHost);

		return dt;
	}

};


// ===========================================================================
//					interop.
// ===========================================================================

void *smallpt_make(unsigned w, unsigned h) {
	return new launcher_t(w, h);
}
void smallpt_destroy(void *smallpt) {
	delete static_cast<launcher_t*>(smallpt);
}

float smallpt_render(void *smallpt, const render_params_t *rp) {
	return static_cast<launcher_t*>(smallpt)->banzai(rp);
}



void cuda_upload_camera(const vec_t cam[4]) {
	cudaMemcpyToSymbol(cu_cam, cam, sizeof(cu_cam));
}
void cuda_upload_scene(const size_t num_spheres, const sphere_t *spheres) {
	cudaMemcpyToSymbol(scene, spheres, sizeof(sphere_t)*num_spheres);
}

