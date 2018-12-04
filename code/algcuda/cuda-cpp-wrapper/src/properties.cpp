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

#include <algcuda/properties.hpp>
#include <algcuda/error.hpp>
#include <cuda_runtime.h>

namespace algcuda {
	namespace device {
		int count() {
			int cnt;
			throw_if_cuda_error(cudaGetDeviceCount(&cnt), "Could not get number of CUDA devices");
			return cnt;
		}

		Id current_default() {
			int cur;
			throw_if_cuda_error(cudaGetDevice(&cur), "Could not get the current CUDA default device");
			return cur;
		}

		int max_threads_per_block() {
			Id dev = current_default();
			return max_threads_per_block(dev);
		}

		namespace cached {
			int max_threads_per_block() {
				static const int result = ::algcuda::device::max_threads_per_block();
				return result;
			}
		}

		int max_threads_per_block(device::Id id) {
			cudaDeviceProp props;
			throw_if_cuda_error(cudaGetDeviceProperties(&props, id), "Could not query device properties");
			return props.maxThreadsPerBlock;
		}
	}
}
