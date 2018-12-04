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

#ifndef ALGCUDA_VECTOR_CUH_INCLUDED_
#define ALGCUDA_VECTOR_CUH_INCLUDED_

#include <algcuda/exit.cuh>
#include <cstddef>

namespace algcuda {
	// provides an unsynchronized vector for device code
	template<typename T> class vector {
	public:
		using size_type = std::ptrdiff_t;

		inline __device__ vector() noexcept :
			M_beg(0), M_end(0), M_cap(0)
		{}

		inline __device__ vector(size_type s) noexcept {
			P_allocate(s);
			for(size_type i = 0; i < s; ++i) {
				new (static_cast<void*>(&M_beg[i])) T();
			}
			M_end = M_beg + s;
		}

		inline __device__ vector(size_type s, const T& v) noexcept {
			P_allocate(s);
			for(size_type i = 0; i < s; ++i) {
				new (static_cast<void*>(&M_beg[i])) T(v);
			}
			M_end = M_beg + s;
		}

		inline __device__ ~vector() noexcept {
			P_destroy();
		}

		inline __device__ vector(const vector& v) noexcept {
			M_beg = M_end = M_cap = 0;

			if(!v.empty()) {
				P_allocate(v.size());
				const T* read_cur = v.M_beg;
				T* write_cur = M_beg;

				while(read_cur < v.M_end) {
					*write_cur = *read_cur;
					++write_cur;
					++read_cur;
				}

				M_end = write_cur;
			}
		}

		inline __device__ vector &operator=(const vector& v) noexcept {
			if(this != &v) {
				P_destroy();
				M_beg = M_end = M_cap = 0;

				if(!v.empty()) {
					P_allocate(v.size());
					const T* read_cur = v.M_beg;
					T* write_cur = M_beg;

					while(read_cur < v.M_end) {
						*write_cur = *read_cur;
						++write_cur;
						++read_cur;
					}

					M_end = write_cur;
				}
			}

			return *this;
		}

		inline __device__ vector(vector&& v) noexcept :
			M_beg(v.M_beg), M_end(v.M_end), M_cap(v.M_cap)
		{
			v.M_beg = v.M_end = v.M_cap = 0;
		}

		inline __device__ vector &operator=(vector&& v) noexcept {
			T* tmp_beg = v.M_beg;
			T* tmp_end = v.M_end;
			T* tmp_cap = v.M_cap;

			v.M_beg = M_beg;
			v.M_end = M_end;
			v.M_cap = M_cap;
			M_beg = tmp_beg;
			M_end = tmp_end;
			M_cap = tmp_cap;

			return *this;
		}

		inline __device__ void resize(size_type s) noexcept {
			T* new_end = M_beg + s;

			if(new_end <= M_end) {
				T* cur = new_end;
				while(cur < M_end) {
					cur->~T();
				}
				M_end = new_end;
			} else {
				if(new_end > M_cap) {
					P_grow(s);
				}

				new_end = M_beg + s;
				while(M_end < new_end) {
					new (static_cast<void*>(M_end)) T();
					++M_end;
				}
			}
		}

		inline __device__ T* release() noexcept {
			T* result = M_beg;
			M_beg = M_end = M_cap = 0;
			return result;
		}

		inline __device__ T* data() noexcept { return M_beg; }
		inline __device__ const T* data() const noexcept { return M_beg; }

		T& __device__ operator[](size_type index) noexcept {
			return M_beg[index];
		}

		const T& __device__ operator[](size_type index) const noexcept {
			return M_beg[index];
		}

		size_type __device__ size() const noexcept {
			return M_end - M_beg;
		}

		bool __device__ empty() const noexcept {
			return M_beg == M_end;
		}

		void __device__ push_back(const T& e) noexcept {
			if(M_end == M_cap) {
				P_grow();
			}

			new (static_cast<void*>(M_end)) T(e);
			++M_end;
		}

		void __device__ push_back(T&& e) noexcept {
			if(M_end == M_cap) {
				P_grow();
			}

			new (static_cast<void*>(M_end)) T(static_cast<T&&>(e));
			++M_end;
		}

		void __device__ pop_back() noexcept {
			--M_end;
			M_end->~T();
		}

		inline __device__ T* begin() noexcept {
			return M_beg;
		}

		inline __device__ T* end() noexcept {
			return M_end;
		}

		inline __device__ const T* begin() const noexcept {
			return M_beg;
		}

		inline __device__ const T* end() const noexcept {
			return M_end;
		}

	private:
		T *M_beg, *M_end, *M_cap;

		void __device__ P_allocate(size_type s) noexcept {
			T* new_buf = static_cast<T*>(malloc(sizeof(T) * s));
			if(!new_buf) {
				trap();
			}

			M_beg = new_buf;
			M_end = new_buf;
			M_cap = new_buf + s;
		}

		void __device__ P_grow() noexcept {
			size_type old_size = size();
			size_type new_size = (old_size + 32) * 2;

			T* new_buf;
			if(new_size <= old_size || !(new_buf = static_cast<T*>(malloc(sizeof(T) * new_size)))) {
				trap();
			}

			T* current_read = M_beg;
			T* current_write = new_buf;
			while(current_read < M_end) {
				*current_write = static_cast<T&&>(*current_read);
				++current_read;
				++current_write;
			}

			free(M_beg);

			M_beg = new_buf;
			M_end = current_write;
			M_cap = new_buf + new_size;
		}

		void __device__ P_grow(size_type new_size) noexcept {
			T* new_buf;
			if(!(new_buf = static_cast<T*>(malloc(sizeof(T) * new_size)))) {
				trap();
			}

			T* current_read = M_beg;
			T* current_write = new_buf;
			while(current_read < M_end) {
				*current_write = static_cast<T&&>(*current_read);
				++current_read;
				++current_write;
			}

			free(M_beg);

			M_beg = new_buf;
			M_end = current_write;
			M_cap = new_buf + new_size;
		}

		void __device__ P_destroy() noexcept {
			if(M_beg) {
				T* current = M_beg;
				while(current < M_end) {
					current->~T();
				}

				free(M_beg);
			}
		}
	};

	// reduce shared memory vectors;
	// call this function with all threads of a block (it calls __syncthreads())
	// and with an array of blockDim.x * blockDim.y vectors, i.e. one vector per thread
	template<int BlockDimX, int BlockDimY, typename T>
		inline __device__ void reduce_thread_vectors(const vector<T>* ptr, vector<T>& output)
	{
		__shared__ std::ptrdiff_t begins[BlockDimX * BlockDimY + 1];
		__syncthreads();

		const int thread_idx = BlockDimX * threadIdx.y + threadIdx.x;
		if(thread_idx == 0) {
			begins[0] = 0;
			for(int i = 0; i < BlockDimX * BlockDimY; ++i) {
				begins[i+1] = begins[i] + ptr[i].size();
			}
			output.resize(begins[BlockDimX * BlockDimY]);
		}

		__syncthreads();

		std::ptrdiff_t my_begin = begins[thread_idx];
		T* o = output.begin() + my_begin;
		const T* i = ptr[thread_idx].begin();
		const T* e = ptr[thread_idx].end();

		for(; i != e; ++i, ++o) {
			*o = *i;
		}

		__syncthreads();
	}
}

#endif //ALGCUDA_VECTOR_HPP
