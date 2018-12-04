// MIT License
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

#include <algcuda/device.hpp>
#include <algcuda/error.hpp>
#include <cuda_runtime.h>

namespace algcuda {
	namespace device {
		void synchronize() {
			throw_if_cuda_error(cudaDeviceSynchronize(), "Device synchronization failed or asynchronous launch failed");
		}

		void synchronize_check_errors() {
			throw_if_cuda_error(cudaGetLastError(), "Synchronous launch failure!");
			throw_if_cuda_error(cudaDeviceSynchronize(), "Device synchronization failed or asynchronous launch produced error");
		}

		void set_printf_buffer_size(std::size_t bytes) {
			throw_if_cuda_error(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, bytes), "Could not set printf buffer size");
		}
	}
}
