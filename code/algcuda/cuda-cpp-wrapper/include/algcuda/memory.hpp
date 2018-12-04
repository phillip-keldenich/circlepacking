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

#ifndef ALGCUDA_MEMORY_HPP_INCLUDED_
#define ALGCUDA_MEMORY_HPP_INCLUDED_

#include <memory>
#include <type_traits>

namespace algcuda {
	namespace device {
		namespace impl {
			extern void do_delete(const void* ptr) noexcept;
			extern void do_device_to_host(const void* from, void* to, std::size_t bytes);
			extern void do_device_to_device(const void* from, void* to, std::size_t bytes);
			extern void do_host_to_device(const void* from, void* to, std::size_t bytes);
			extern void* do_device_alloc(std::size_t bytes);
		}

		template<typename T> struct Device_deleter {
			void operator()(T* ptr) const noexcept {
				if(ptr) {
					ptr->~T();
					impl::do_delete(ptr);
				}
			}
		};

		template<typename T> struct Device_deleter<T[]> {
			void operator()(T* ptr) const noexcept {
				static_assert(std::is_trivially_destructible<T>::value, "Device_deleter can only be used for arrays of trivially destructible objects!");
				if(ptr) {
					impl::do_delete(ptr);
				}
			}
		};

		template<typename T> struct Memory_t {
			using type = std::unique_ptr<T, Device_deleter<const T>>;
		};

		template<typename T> struct Memory_t<T[]> {
			using type = std::unique_ptr<T[], Device_deleter<const T[]>>;
		};

		template<typename T> using Memory = typename Memory_t<T>::type;

		template<typename T> inline void copy(const Memory<T[]>& from, std::size_t count, T* to) {
			impl::do_device_to_host(static_cast<const void*>(from.get()), static_cast<void*>(to), count * sizeof(T));
		}

		template<typename T> inline void copy(const Memory<T>& from, T* to) {
			impl::do_device_to_host(static_cast<const void*>(from.get()), static_cast<void*>(to), sizeof(T));
		}

		template<typename T> inline void copy(const Memory<T[]>& from, std::size_t count, const Memory<T[]>& to) {
			impl::do_device_to_device(static_cast<const void*>(from.get()), static_cast<void*>(to.get()), count * sizeof(T));
		}

		template<typename T> inline void copy(const Memory<T>& from, const Memory<T>& to) {
			impl::do_device_to_device(static_cast<const void*>(from.get()), static_cast<void*>(to.get()), sizeof(T));
		}

		template<typename T> inline void copy(const T* from, std::size_t count, const Memory<T[]>& to) {
			impl::do_host_to_device(static_cast<const void*>(from), static_cast<void*>(to.get()), count * sizeof(T));
		}

		template<typename T> inline void copy(const T* from, const Memory<T>& to) {
			impl::do_host_to_device(static_cast<const void*>(from), static_cast<void*>(to.get()), sizeof(T));
		}

		template<typename T> inline Memory<T> make_unique() {
			static_assert(std::is_pod<T>::value, "make_unique for device memory only works with POD (plain old data) types!");
			void* ptr = impl::do_device_alloc(sizeof(T));
			return Memory<T>{static_cast<T*>(ptr)};
		}

		template<typename T> inline Memory<typename std::enable_if<!std::is_array<T>::value,T>::type> make_unique(const T& v) {
			static_assert(std::is_pod<T>::value, "make_unique for device memory only works with POD (plain old data) types!");
			void* ptr = impl::do_device_alloc(sizeof(T));
			Memory<T> result{static_cast<T*>(ptr)};
			copy(&v, result);
			return result;
		}

		template<typename T> inline Memory<typename std::enable_if<std::is_array<T>::value,T>::type> make_unique(std::size_t s) {
			using R = typename std::remove_extent<T>::type;
			static_assert(std::is_pod<R>::value, "make_unique for device memory only works with POD (plain old data) types!");
			void* ptr = impl::do_device_alloc(sizeof(R) * s);
			return Memory<T>{static_cast<R*>(ptr)};
		}
	}
}

#endif

