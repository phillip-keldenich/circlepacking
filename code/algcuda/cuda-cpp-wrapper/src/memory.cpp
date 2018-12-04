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

#include <cuda_runtime_api.h>
#include <algcuda/error.hpp>
#include <algcuda/memory.hpp>

void algcuda::device::impl::do_delete(const void* ptr) noexcept {
	throw_if_cuda_error(cudaFree(const_cast<void*>(ptr)), "Fatal - failed to free device memory");
}

void algcuda::device::impl::do_device_to_host(const void* from, void* to, std::size_t bytes) {
	throw_if_cuda_error(cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToHost), "Could not copy device memory to host");
}

void algcuda::device::impl::do_device_to_device(const void* from, void* to, std::size_t bytes) {
	throw_if_cuda_error(cudaMemcpy(to, from, bytes, cudaMemcpyDeviceToDevice), "Could not copy device memory to device");
}

void algcuda::device::impl::do_host_to_device(const void* from, void* to, std::size_t bytes) {
	throw_if_cuda_error(cudaMemcpy(to, from, bytes, cudaMemcpyHostToDevice), "Could not copy host memory to device");
}

void* algcuda::device::impl::do_device_alloc(std::size_t bytes) {
	void* result;
	throw_if_cuda_error(cudaMalloc(&result, bytes), "Failed to allocate device memory");
	return result;
}

