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

#ifndef RUN_KERNEL_CUH_INCLUDED_
#define RUN_KERNEL_CUH_INCLUDED_

#include "operations.cuh"

#include <algcuda/device.hpp>
#include <algcuda/memory.hpp>
#include <algcuda/atomic_float.hpp>

#include <queue>
#include <vector>
#include <csignal>
#include <cstdlib>

namespace circlepacking {
	static sig_atomic_t was_interrupted = 0;
	static void (*store_handler)(int);

	static void interrupt_handler(int) {
		if(was_interrupted) {
			std::signal(SIGINT, store_handler);
			std::raise(SIGINT);
			std::abort();
		}

		was_interrupted = 1;
	}

	static void start_interruptible_computation() {
		was_interrupted = 0;
		store_handler = std::signal(SIGINT, &interrupt_handler);
	}

	static void stop_interruptible_computation() {
		std::signal(SIGINT, store_handler);
		store_handler = SIG_DFL;
	}

	template<typename DeviceCallable> static inline __host__ bool run_kernel_until_no_criticals(DeviceCallable call, IV outer_la, dim3 grid, dim3 block) {
		using algcuda::Atomic_ordered_double;

		auto d_first_problematic = algcuda::device::make_unique<Atomic_ordered_double>();
		Atomic_ordered_double h_first_problematic;

		struct Comp_lower {
			inline bool operator()(const IV& i1, const IV& i2) const noexcept { return i1.get_lb() > i2.get_lb(); }
		};

		std::priority_queue<IV, std::vector<IV>, Comp_lower> q;
		q.push(outer_la);

		start_interruptible_computation();
		while(!q.empty()) {
			IV current = q.top();
			q.pop();

			std::cerr << "Handling lambda interval " << current << std::endl;

			h_first_problematic.set(DBL_MAX);
			algcuda::device::copy(&h_first_problematic, d_first_problematic);
			call<<<grid,block>>>(current, d_first_problematic.get(), false);
			algcuda::device::copy(d_first_problematic, &h_first_problematic);

			if(was_interrupted) {
				was_interrupted = 0;
				IV last_run;

				if(h_first_problematic.get() != DBL_MAX) {
					last_run = current;
				} else {
					if(q.empty()) {
						stop_interruptible_computation();
						return true;
					}

					last_run = q.top();
				}

				std::cerr << "We were interrupted - running the first remaining interval with output once!" << std::endl;
				h_first_problematic.set(DBL_MAX);
				algcuda::device::copy(&h_first_problematic, d_first_problematic);
				call<<<grid,block>>>(last_run, d_first_problematic.get(), true);
				algcuda::device::copy(d_first_problematic, &h_first_problematic);
				stop_interruptible_computation();
				return false;
			}

			if(h_first_problematic.get() != DBL_MAX) {
				double width_before = current.get_ub() - current.get_lb();
				double width_after = current.get_ub() - h_first_problematic.get();

				if(width_before >= 2*width_after) {
					// just continue with new interval
					q.push(IV(h_first_problematic.get(), current.get_ub()));
				} else {
					// split interval
					double mid = 0.5 * (h_first_problematic.get() + current.get_ub());
					if(h_first_problematic.get() < mid) {
						q.push(IV(h_first_problematic.get(), mid));
					}
					if(mid <= current.get_ub()) {
						q.push(IV(mid, current.get_ub()));
					}
				}
			}
		}
		stop_interruptible_computation();

		return true;
	}
}

#endif

