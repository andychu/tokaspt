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
	Large portions were lifted one way or another from http://www-old.amolf.nl/~vanmeel/mdgpu/
	Copyright (c) 2007 A. Arnold and J. A. van Meel, FOM institute AMOLF, Amsterdam; all rights reserved unless otherwise stated.
	I AM REQUIRED TO CITE THE ARTICLE WHILE STANDING ON ONE FEET: 
		"Harvesting graphics power for MD simulations" 
		by J.A. van Meel, A. Arnold, D. Frenkel, S. F. Portegies Zwart and R. G. Belleman, arXiv:0709.3225.
*/
#ifndef PRNG_H
#define PRNG_H

#include "specifics.h"
#include "rt_alloc.h" // cuda_allocate

//BITROT: dive in there again and clear it up.

// nvcc isn't too fond of namespaces; let's get medieval.
// we have 3 prng: online(1), baked(2), rigged.
#define USE_RNG 2

#if USE_RNG == 2
	#define RNG_IS_BAKED
	#define RNG_BAKED_USES_SHARED	// trade 4 bytes/thread of shared mem to get down to 24 registers.
	#if __DEVICE_EMULATION__		// note: device emu fails when it's turned on. i don't care.
		#undef RNG_BAKED_USES_SHARED
	#endif
#endif



#define MAGIC_A 0x5DEECE66DLL
#define MAGIC_C 0xB
#define SPLIT24(val) make_uint2( ((val) & 0xFFFFFFLL), ((val >> 24) & 0xFFFFFFLL) )

// #ifdef RNG_IS_BAKED
	static __global__ void bake_some_kernel(uint2 A, uint2 C, unsigned num_chunks, uint2 *states, float *data);
//#endif

namespace rng_details {
	enum { mantissa_mask = (1u<<23)-1 };
	// bugs. more bugs. pfff.
	static __device__ float to01(const uint32_t bits) { return __int_as_float(bits|0x3f800000u) - 1.f; }


	namespace rand48 {
		// magic constants for rand48
		// static const unsigned long long a = 0x5DEECE66DLL, c = 0xB;

		enum { is_baked = 0 };
		typedef uint2 state_t;
		typedef void data_t;

		static __device__ float lcg(const uint2 A, const uint2 C, const state_t &in, state_t &out) {
			// low 24-bit multiplication
			// uint32_t R0, R1;
			const uint32_t lo0 = __umul24(in.x, A.x);
			const uint32_t lo1 = __umulhi(in.x, A.x);
			// cross-terms, low/hi 24-bit multiplication
			const uint32_t cr0 = __umul24(in.x, A.y);
			const uint32_t cr1 = __umul24(in.y, A.x);

			uint32_t R0 = (lo0 & 0xFFFFFF) + C.x;
			uint32_t R1 = cr0 + cr1 + C.y;
			R1 += ((lo0 >> 24) | (lo1 << 8)) + (R0 >> 24); /* overflow */

			R0 &= 0xFFFFFF;
			R1 &= 0xFFFFFF;

			uint32_t bits = R1 >> 1;
			float result = to01(bits);

			out = make_uint2(R0, R1);
			return result;
		}

		static __device__ float next(const state_t &in, state_t &out) {
			const uint2 A(SPLIT24(MAGIC_A));
			const uint2 C(SPLIT24(MAGIC_C));

			// low 24-bit multiplication
			// uint32_t R0, R1;
			const uint32_t lo0 = __umul24(in.x, A.x);
			const uint32_t lo1 = __umulhi(in.x, A.x);
			// cross-terms, low/hi 24-bit multiplication
			const uint32_t cr0 = __umul24(in.x, A.y);
			const uint32_t cr1 = __umul24(in.y, A.x);

			uint32_t R0 = (lo0 & 0xFFFFFF) + C.x;
			uint32_t R1 = cr0 + cr1 + C.y;
			R1 += ((lo0 >> 24) | (lo1 << 8)) + (R0 >> 24); /* overflow */

			R0 &= 0xFFFFFF;
			R1 &= 0xFFFFFF;

			uint32_t bits = R1 >> 1;
			float result = to01(bits);

			out = make_uint2(R0, R1);
			return result;
		}

		#if 0
			// eats too many regs.
			struct proxy_t {
				state_t shadow;
				__device__ proxy_t(const state_t *src) : shadow(*src) {}
				__device__ void commit(state_t *dst) const { *dst = shadow; }
				__device__ float2 gen2() { return make_float2(next(shadow, shadow), next(shadow, shadow));  }
				__device__ float3 gen3() { return make_float3(next(shadow, shadow), next(shadow, shadow), next(shadow, shadow)); }
			};
		#endif

		// for this PRNG we'll avoid touching global mem as much as possible.
		// saves only some read/write but costs 0 regs.
		template<block_size_t block_size>
			struct proxy_t {
				state_t &s;
				__device__ proxy_t(state_t *src) : s(*src) {}
				// buffered
				__device__ float2 gen2() { 
					state_t shadow;
					float2 r(make_float2(next(s, shadow), next(shadow, shadow)));
					s = shadow;
					return r;
				}
				__device__ float3 gen3() { 
					state_t shadow;
					float3 r(make_float3(next(s, shadow), next(shadow, shadow), next(shadow, shadow)));
					s = shadow;
					return r;
				}
			};

		//FIXME: tile_id, thread_id
		static __device__ void seed(state_t &s, const uint32_t sx, const uint32_t sy) {
			uint64_t x = sx, y = sy;
			uint64_t mix1;
			if (0) {
				x ^= y;
				x *= sx;
				y += x;
				mix1 = (y << 16) | x;
			}
			else {
				x = 69069*x + 12345;
				y ^= y<<13;
				y ^= y>>17;
				y ^= y<< 5;
				mix1 = x + y;
			}

			
			uint64_t mix2 = (mix1 << 16) | 0x330E;
			uint64_t iter = MAGIC_A*mix2 + MAGIC_C;
			s = SPLIT24(iter);
		}
	} // rand48

	namespace baked {
		enum { is_baked = 1 };
		// beware, states aren't meant to be visible to rendering kernels.
		typedef uint2 state_t;
		typedef float data_t;


		// alternatively, store the pointer in shared mem to get 1 free reg and warning ;)
		//	'Advisory: Cannot tell what pointer points to, assuming global memory space'
		// assumes it cannot ever exhaust that pool.
		template<block_size_t block_size>
		struct proxy_t {
			#ifdef RNG_BAKED_USES_SHARED
				const data_t *&base;
				__device__ proxy_t(const data_t *&src) : base(src){}
			#else
				const data_t *base;
				__device__ proxy_t(const data_t *src) : base(src){}
			#endif

			__device__ float2 gen2() {
				base += block_size*2;
				return make_float2(*(base - block_size*2), *(base - block_size*1));
			}
			__device__ float3 gen3() { 
				base += block_size*3;
				return make_float3(*(base - block_size*3), *(base - block_size*2), *(base - block_size*1));
			}
		};

		static __device__ void seed(state_t &s, const uint32_t tile_id, const uint32_t thread_id) {}


		// it's not what we'd like... we want each of our rendering thread to have its own generator;
		// here we have 1 parallelized generator. anyway.
		enum { param_threads = 192, param_blocks = 32, param_n = param_threads*param_blocks };
		// with 10 registers, it seems we need 256 to theoritically get 100% occupancy.
		// enum { param_threads = 256, param_blocks = 128, param_n = param_threads*param_blocks };
		// enum { param_threads = 256, param_blocks = 64, param_n = param_threads*param_blocks };
		// rounded n# of chunks.
		static unsigned compute_num_chunks(unsigned want) { return (want + param_n - 1) / param_n; }
		// allocation helper
		static size_t compute_data_size(unsigned want) { return sizeof(data_t)*compute_num_chunks(want)*param_n; }
		// computed stridified LCG params.
		static uint2 A, C;
		// num_stream should be used for seeding
		//FIXME: we need more than 1 stream to decorellate a bit.
		static void init_baked_rng(const unsigned id, uint2 *&states) {
			state_t seeds[param_n];
			cuda_allocate<true>(states, param_n);

			// calculate strided iteration constants
			uint64_t a = 1LL, c = 0;
			for (size_t i = 0; i < param_n; ++i) {
				c += MAGIC_C*a;
				a *= MAGIC_A;
			}
			A = SPLIT24(a);
			C = SPLIT24(c);

			// prepare first nThreads random numbers from seed
			uint64_t main_seed = 0x79dedea3*(id + 1);
			uint64_t x = (main_seed << 16) | 0x330E;
			for (unsigned i = 0; i<param_n; ++i) {
				x = MAGIC_A*x + MAGIC_C;
				seeds[i] = SPLIT24(x);
			}
			CUDA_SAFE_CALL(cudaMemcpy(states, seeds, sizeof(state_t)*param_n, cudaMemcpyHostToDevice));
		}


		// generate 'want' randoms.
		static void bake_some(cudaStream_t stream, unsigned want, state_t *states, data_t *data) {
			unsigned num_chunks = compute_num_chunks(want);
			bake_some_kernel<<<param_blocks, param_threads, 0, stream>>>(A, C, num_chunks, states, data);
			if (check_kernel_calls) cutilCheckMsg("'bake_some_kernel' kernel failure\n");
		}
	}

	namespace rigged {
		// fake & minimal.
		enum { is_baked = 0 };
		typedef char state_t;
		typedef void data_t;

		template<block_size_t block_size>
			struct proxy_t {
				__device__ proxy_t(const state_t *) {}
				__device__ float2 gen2() const { return make_float2(0.5f, 0.5f); }
				__device__ float3 gen3() const { return make_float3(.125f, 0.25f, 0.75f); }
			};

		static __device__ void seed(state_t &s, const uint32_t tile_id, const uint32_t thread_id) {}
	}
} // rng_details

#ifdef RNG_IS_BAKED
	// generate 1 stream of randoms. 
	void bake_some_kernel(uint2 A, uint2 C, unsigned num_chunks, uint2 *states, float *data) {
		const uint32_t n = gridDim.x*blockDim.x;
		const uint32_t tid = blockDim.x*blockIdx.x + threadIdx.x;
		uint2 state = states[tid];
		float *p = data + tid;
		unsigned i = num_chunks;
		do {
			//FIXME: first state has already been generated.
			*p = rng_details::rand48::lcg(A, C, state, state);
			p += n;
		} while (--i);
		states[tid] = state;
	}
#else
	void bake_some_kernel(uint2 A, uint2 C, unsigned num_chunks, uint2 *states, float *data) {}
#endif


#undef MAGIC_A
#undef MAGIC_C
#undef SPLIT24

#if USE_RNG == 1
	namespace rng = rng_details::rand48;
#elif USE_RNG == 2
	namespace rng = rng_details::baked;
#else
	namespace rng = rng_details::rigged;
#endif

#endif
