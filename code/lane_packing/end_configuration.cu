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

#include "operations.cuh"
#include "../subintervals.hpp"
#include <algcuda/device.hpp>
#include <algcuda/exit.cuh>
#include "run_kernel.cuh"

namespace circlepacking {
	namespace lane_packing {
		void find_critical_intervals_end_configuration();
	}
}

using namespace circlepacking;
using namespace circlepacking::lane_packing;
using algcuda::Atomic_ordered_double;

constexpr int la_subintervals_2d = 4096;
constexpr int r1_subintervals_2d = 1024;
constexpr int r2_subintervals_2d = 1024;
constexpr int la_parallel_2d = 256;
constexpr int r1_parallel_2d = 256;
constexpr int r2_parallel_2d = 256;

constexpr int la_subintervals_3d = 1024;
constexpr int r1_subintervals_3d = 256;
constexpr int r2_subintervals_3d = 128;
constexpr int r3_subintervals_3d = 128;
constexpr int la_parallel_3d = 256;
constexpr int r1_parallel_3d = 128;
constexpr int r2_parallel_3d = 16;
constexpr int r3_parallel_3d = 16;

static inline __device__ IV compute_r1_range(IV la) {
	// any disk above this can simply be placed on its own (we assume the disks fit into the lane)
	double ub_by_single_placement = lower_bound_radius_single_placement(la.get_lb());
	double ub_by_lane_width = (0.5*(1.0-la)).get_ub();
	
	// any disk below this would immediately open a new lane
	double lb_by_lane_width = (0.25*(1.0-la)).get_lb();
	return {lb_by_lane_width, ub_by_lane_width < ub_by_single_placement ? ub_by_lane_width : ub_by_single_placement};
}

static inline __device__ IV compute_r2_range(IV la, IV r1) {
	double lb_lane_width_r1 = __dmul_rd(0.5, __dadd_rd(1.0, -__dadd_ru(la.get_ub(), __dmul_ru(2.0, r1.get_ub()))));
	if(lb_lane_width_r1 < 0.0) { lb_lane_width_r1 = 0.0; }
	return {lb_lane_width_r1, r1.get_ub()};
}

static __device__ bool handle_inner_outer_2d(IV la, IV r1, IV r2) {
	IV a = collision_angle_inner_outer(la, r1, r2) + semicircle_angle_outer(r2);
	double ub_lane_container_area_ratio = __dadd_ru(1.0, -__dmul_rd(la.get_lb(), la.get_lb()));
	double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, a.get_ub()));
	
	IV rho = (1.0 - 2.0*r2).square() - la.square();
	rho.tighten_lb(0.0);
	double lb_alpha_reusable = (a - semicircle_angle_inner(la,r1)).get_lb();
	double lb_area_reusable = __dmul_rd(__dmul_rd(0.5, lb_alpha_reusable), rho.get_lb());

	double lb_area1 = __dmul_rd(lb_pi, __dmul_rd(r1.get_lb(), r1.get_lb()));
	double lb_area2 = __dmul_rd(lb_pi, __dmul_rd(r2.get_lb(), r2.get_lb()));
	double lb_disk_area = __dadd_rd(lb_area2, __dmul_rd(0.5, lb_area1));
	return lb_disk_area >= __dmul_ru(critical_ratio, __dadd_ru(ub_area_used, -lb_area_reusable));
}

static __device__ bool handle_outer_inner_2d(IV la, IV r1, IV r2) {
	IV a = collision_angle_outer_inner(la, r1, r2) + semicircle_angle_inner(la, r2);
	double ub_lane_container_area_ratio = __dadd_ru(1.0, -__dmul_rd(la.get_lb(), la.get_lb()));
	double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, a.get_ub()));
		
	IV rho = 1.0 - (la + 2.0*r2).square();
	rho.tighten_lb(0.0);
	double lb_alpha_reusable = (a-semicircle_angle_outer(r1)).get_lb();
	double lb_area_reusable = __dmul_rd(__dmul_rd(0.5, lb_alpha_reusable), rho.get_lb());

	double lb_area1 = __dmul_rd(lb_pi, __dmul_rd(r1.get_lb(), r1.get_lb()));
	double lb_area2 = __dmul_rd(lb_pi, __dmul_rd(r2.get_lb(), r2.get_lb()));
	double lb_disk_area = __dadd_rd(lb_area2, __dmul_rd(0.5, lb_area1));
	return lb_disk_area >= __dmul_ru(critical_ratio, __dadd_ru(ub_area_used, -lb_area_reusable));
}

static __global__ void kernel_find_critical_intervals_end_configuration_2d(IV outer_la, Atomic_ordered_double* first_problematic_la, bool output_criticals) {
	for(int la_offset = blockIdx.x; la_offset < la_subintervals_2d; la_offset += gridDim.x) {
		IV la = get_subinterval(outer_la, la_offset, la_subintervals_2d);
		IV r1_range = compute_r1_range(la);
		for(int r1_offset = blockIdx.y; r1_offset < r1_subintervals_2d; r1_offset += gridDim.y) {
			IV r1_ = get_subinterval(r1_range, r1_offset, r1_subintervals_2d);
			IV r2_range = compute_r2_range(la, r1_);
			for(int r2_offset = threadIdx.x; r2_offset < r2_subintervals_2d; r2_offset += blockDim.x) {
				IV r2 = get_subinterval(r2_range, r2_offset, r2_subintervals_2d);
				IV r1 = r1_;
				r1.tighten_lb(r2.get_lb());

				if(__dmul_ru(2.0, r1.get_ub()) >= __dadd_rd(1.0, -la.get_ub())) {
					printf("End configuration (2 disks): Circle possibly does not fit - la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g]\n",
						la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub()
					);
					algcuda::trap();
				}

				if(!handle_inner_outer_2d(la, r1, r2)) {
					if(output_criticals) {
						printf("End configuration (2 disks): Critical interval with r1 on the inside - la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g]\n",
							la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub()
						);
					} else {
						first_problematic_la->store_min(la.get_lb());
						return;
					}
				}

				if(!handle_outer_inner_2d(la, r1, r2)) {
					if(output_criticals) {
						printf("End configuration (2 disks): Critical interval with r1 on the outside - la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g]\n",
							la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub()
						);
					} else {
						first_problematic_la->store_min(la.get_lb());
						return;
					}
				}
			}
		}
	}
}

static inline __device__ IV compute_r3_range_3d(IV la, IV r1, IV r2) {
	double lb_lane_width_r2 = __dmul_rd(0.5, __dadd_rd(1.0, -__dadd_ru(la.get_ub(), __dmul_ru(2.0, r2.get_ub()))));
	if(lb_lane_width_r2 < 0.0) { lb_lane_width_r2 = 0.0; }
	return {lb_lane_width_r2, r2.get_ub()};
}

static inline __device__ bool handle_r1_inner_3d(IV la, IV r1, IV r2, IV r3) {
	IV a12 = collision_angle_inner_outer(la, r1, r2);
	IV a23 = collision_angle_outer_inner(la, r2, r3);
	IV a13 = collision_angle_inner_inner(la, r1, r3);

	if(definitely(a12 + a23 >= a13)) {
		// we already handle this case as 2 times 2 semicircles
		return true;
	}

	IV a3 = semicircle_angle_inner(la, r3);
	IV a = a13+a3;
	IV a2 = semicircle_angle_outer(r2);
	IV reusable_alpha = a - (a12+a2);

	IV rho = 1.0 - (la + 2.0*r3).square();
	rho.tighten_lb(0.0);
	double lb_area_reusable = __dmul_rd(__dmul_rd(0.5, reusable_alpha.get_lb()), rho.get_lb());

	double ub_lane_container_area_ratio = __dadd_ru(1.0, -__dmul_rd(la.get_lb(), la.get_lb()));
	double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, a.get_ub()));
	double lb_area1 = __dmul_rd(lb_pi, __dmul_rd(r1.get_lb(), r1.get_lb()));
	double lb_area2 = __dmul_rd(lb_pi, __dmul_rd(r2.get_lb(), r2.get_lb()));
	double lb_area3 = __dmul_rd(lb_pi, __dmul_rd(r3.get_lb(), r3.get_lb()));

	double lb_disk_area = __dadd_rd(__dadd_rd(lb_area2, lb_area3), __dmul_rd(0.5, lb_area1));
	return lb_disk_area >= __dmul_ru(critical_ratio, __dadd_ru(ub_area_used, -lb_area_reusable));
}

static inline __device__ bool handle_r1_outer_3d(IV la, IV r1, IV r2, IV r3) {
	IV a12 = collision_angle_outer_inner(la, r1, r2);
	IV a23 = collision_angle_inner_outer(la, r2, r3);
	IV a13 = collision_angle_outer_outer(r1, r3);

	if(definitely(a12+a23 >= a13)) {
		// we already handle this case as 2 tiems 2 semicircles
		return true;
	}

	IV a3 = semicircle_angle_outer(r3);
	IV a = a13+a3;
	IV a2 = semicircle_angle_inner(la, r2);
	IV reusable_alpha = a - (a12+a2);

	IV rho = (1.0 - 2.0*r3).square() - la.square();
	rho.tighten_lb(0.0);
	double lb_area_reusable = __dmul_rd(__dmul_rd(0.5, reusable_alpha.get_lb()), rho.get_lb());

	double ub_lane_container_area_ratio = __dadd_ru(1.0, -__dmul_rd(la.get_lb(), la.get_lb()));
	double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, a.get_ub()));
	double lb_area1 = __dmul_rd(lb_pi, __dmul_rd(r1.get_lb(), r1.get_lb()));
	double lb_area2 = __dmul_rd(lb_pi, __dmul_rd(r2.get_lb(), r2.get_lb()));
	double lb_area3 = __dmul_rd(lb_pi, __dmul_rd(r3.get_lb(), r3.get_lb()));

	double lb_disk_area = __dadd_rd(__dadd_rd(lb_area2, lb_area3), __dmul_rd(0.5, lb_area1));
	return lb_disk_area >= __dmul_ru(critical_ratio, __dadd_ru(ub_area_used, -lb_area_reusable));
}

static __global__ void kernel_find_critical_intervals_end_configuration_3d(IV outer_la, Atomic_ordered_double* first_problematic_la, bool output_criticals) {
	for(int la_offset = blockIdx.x; la_offset < la_subintervals_3d; la_offset += gridDim.x) {
		IV la = get_subinterval(outer_la, la_offset, la_subintervals_3d);
		IV r1_range = compute_r1_range(la);
		for(int r1_offset = blockIdx.y; r1_offset < r1_subintervals_3d; r1_offset += gridDim.y) {
			IV r1_ = get_subinterval(r1_range, r1_offset, r1_subintervals_3d);
			IV r2_range = compute_r2_range(la, r1_);
			for(int r2_offset = threadIdx.x; r2_offset < r2_subintervals_3d; r2_offset += blockDim.x) {
				IV r2_ = get_subinterval(r2_range, r2_offset, r2_subintervals_3d);
				IV r3_range = compute_r3_range_3d(la, r1_, r2_);
				if(r3_range.empty()) {
					continue;
				}

				for(int r3_offset = threadIdx.y; r3_offset < r3_subintervals_3d; r3_offset += blockDim.y) {
					IV r3 = get_subinterval(r3_range, r3_offset, r3_subintervals_3d);
					IV r2 = r2_;
					r2.tighten_lb(r3.get_lb());
					IV r1 = r1_;
					r1.tighten_lb(r2.get_lb());
					if(r2.empty() || r1.empty()) {
						continue;
					}

					if(__dmul_ru(2.0, r1.get_ub()) >= __dadd_rd(1.0, -la.get_ub())) {
						printf("End configuration (3 disks): Circle possibly does not fit - la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g], r3 = [%.19g,%.19g]\n",
							la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub(), r3.get_lb(), r3.get_ub()
						);
						algcuda::trap();
					}

					if(!handle_r1_inner_3d(la, r1, r2, r3)) {
						if(output_criticals) {
							printf("End configuration (3 disks): Critical interval with r1 on the inside - la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g], r3 = [%.19g,%.19g]\n",
								la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub(), r3.get_lb(), r3.get_ub()
							);
						} else {
							first_problematic_la->store_min(la.get_lb());
							return;
						}
					}

					if(!handle_r1_outer_3d(la, r1, r2, r3)) {
						if(output_criticals) {
							printf("End configuration (3 disks): Critical interval with r1 on the outside - la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g], r3 = [%.19g,%.19g]\n",
								la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub(), r3.get_lb(), r3.get_ub()
							);
						} else {
							first_problematic_la->store_min(la.get_lb());
							return;
						}
					}
				}
			}
		}
	}
}

static void find_critical_intervals_end_configuration_2d() {
	const dim3 grid(la_parallel_2d, r1_parallel_2d);
	std::cout << "End configuration (2 disks) ..." << std::endl;
	run_kernel_until_no_criticals(&kernel_find_critical_intervals_end_configuration_2d, IV(lambda_min, lambda_max), grid, dim3(r2_parallel_2d));
}

static void find_critical_intervals_end_configuration_3d() {
	const dim3 grid(la_parallel_3d, r1_parallel_3d);
	const dim3 block(r2_parallel_3d, r3_parallel_3d);
	std::cout << "End configuration (3 disks) ..." << std::endl;
	run_kernel_until_no_criticals(&kernel_find_critical_intervals_end_configuration_3d, IV(lambda_min, lambda_max), grid, block);
}

void circlepacking::lane_packing::find_critical_intervals_end_configuration() {
	find_critical_intervals_end_configuration_2d();
	find_critical_intervals_end_configuration_3d();
}

