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

#include <algcuda/error.hpp>
#include <cuda_runtime.h>
#include <sstream>
#include <cstring>

std::string algcuda::Cuda_category::message(int ev) const {
	const char* cerr_name = cudaGetErrorName(static_cast<cudaError_t>(ev));
	const char* cerr_str  = cudaGetErrorName(static_cast<cudaError_t>(ev));

	if(!cerr_name || !cerr_str) {
		return "Unknown CUDA error!";
	}

	if(!std::strstr(cerr_str, cerr_name)) {
		std::ostringstream format;
		format << cerr_name << ": " << cerr_str;
		return format.str();
	} else {
		return cerr_str;
	}
}

void algcuda::last_error::clear() noexcept {
	cudaGetLastError();
}

int algcuda::last_error::get_and_clear() noexcept {
	return cudaGetLastError();
}

int algcuda::last_error::get() noexcept {
	return cudaPeekAtLastError();
}
