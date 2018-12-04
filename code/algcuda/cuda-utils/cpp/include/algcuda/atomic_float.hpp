//MIT License
//
//Copyright (c) 2018 TU Braunschweig, Algorithms Group
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.

#ifndef ALGCUDA_ATOMIC_FLOAT_HPP_INCLUDED_
#define ALGCUDA_ATOMIC_FLOAT_HPP_INCLUDED_

#include <cstdint>
#include <algcuda/macros.hpp>

namespace algcuda {
	class Atomic_ordered_double {
	public:
		__host__ __device__ void set(double d) noexcept {
			value = transform(d);
		}

		__host__ __device__ double get() const noexcept {
			return inverse_transform(value);
		}

#ifdef __CUDACC__
		__device__ void store_max(double v) noexcept {
			atomicMax(&value, transform(v));
		}

		__device__ void store_min(double v) noexcept {
			atomicMin(&value, transform(v));
		}
#endif

	private:
		static_assert(sizeof(unsigned long long) == sizeof(double));

		using ad = __attribute__((__may_alias__)) double;
		using au = __attribute__((__may_alias__)) unsigned long long;

		static inline __host__ __device__ unsigned long long transform(double value) noexcept {
			unsigned long long vv = *reinterpret_cast<au*>(&value);
			unsigned long long mask = -(vv >> 63) | (unsigned long long)(1) << 63;
			return vv ^ mask;
		}

		static inline __host__ __device__ double inverse_transform(unsigned long long value) noexcept {
			unsigned long long mask = ((value >> 63) - 1) | (unsigned long long)(1) << 63;
			unsigned long long vv = value ^ mask;
			return *reinterpret_cast<ad*>(&vv);
		}

		unsigned long long value;
	};
}

#endif

