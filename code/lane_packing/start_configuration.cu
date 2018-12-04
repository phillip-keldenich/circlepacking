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
#include "run_kernel.cuh"
#include "../subintervals.hpp"
#include <algcuda/device.hpp>
#include <algcuda/memory.hpp>
#include <algcuda/atomic_float.hpp>

namespace circlepacking {
	namespace lane_packing {
		void find_critical_intervals_start_configuration();
		void find_critical_intervals_start_end_configuration();
	}
}

using namespace circlepacking;
using namespace circlepacking::lane_packing;
using algcuda::Atomic_ordered_double;

static constexpr int la_subintervals = 256;
static constexpr int r1_subintervals = 256;
static constexpr int r2_subintervals = 256;
static constexpr int r3_subintervals = 64;

static constexpr int la_parallel = 128;
static constexpr int r1_parallel = 64;
static constexpr int r2_parallel = 32;
static constexpr int r3_parallel = 16;

static constexpr int la_subintervals_2d = 4096;
static constexpr int r1_subintervals_2d = 4096;
static constexpr int r2_subintervals_2d = 4096;
static constexpr int la_parallel_2d = 512;
static constexpr int r1_parallel_2d = 512;
static constexpr int r2_parallel_2d = 256;

static inline __device__ IV compute_r1_range(IV la) {
	// any disk above this can simply be placed on its own (we assume the disks fit into the lane)
	double ub_by_single_placement = lower_bound_radius_single_placement(la.get_lb());
	double ub_by_lane_width = (0.5*(1.0-la)).get_ub();
	
	// any disk below this would immediately open a new lane
	double lb_by_lane_width = (0.25*(1.0-la)).get_lb();
	return {lb_by_lane_width, ub_by_lane_width < ub_by_single_placement ? ub_by_lane_width : ub_by_single_placement};
}

static inline __device__ IV compute_r2_range(IV la, IV r1) {
	double lb_by_lane_width = (0.25*(1.0-la)).get_lb();
	return {lb_by_lane_width, r1.get_ub()};
}

// in the two-disk case, we only know that r2 must not fit past r1
static inline __device__ IV compute_r2_range_2d(IV la, IV r1) {
	double lb_by_lane_width = (0.5*(1.0-la-2.0*r1)).get_lb();
	return {lb_by_lane_width, r1.get_ub()};
}

static inline __device__ IV compute_r3_range(IV la, IV r1, IV r2) {
	double lb_by_lane_width = (0.25*(1.0-la)).get_lb();
	return {lb_by_lane_width, r2.get_ub()};
}

static inline __device__ IV compute_r3_range_3d(IV la, IV r1, IV r2) {
	double lb_by_lane_width = (0.5*(1.0-la-2.0*r2)).get_lb();
	return {lb_by_lane_width, r2.get_ub()};
}

// in the start configuration, r1 is the first disk that is not placed on its own,
// and r3 is NOT the last disk of the strip. We start by placing r1 on the outside;
// there are two cases: (1) r3 is stopped by r2, or (2) r3 is stopped r1.

static __device__ bool handle_case_1(IV la, IV r1, IV r2, IV a12) {
	double ub_lane_container_area_ratio = __dadd_ru(1.0, -__dmul_rd(la.get_lb(), la.get_lb()));
	double lb_area1 = __dmul_rd(lb_pi, __dmul_rd(r1.get_lb(), r1.get_lb()));
	double lb_area2 = __dmul_rd(lb_pi, __dmul_rd(r2.get_lb(), r2.get_lb()));
	IV a = semicircle_angle_outer(r1) + a12;
	double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, a.get_ub()));
	double lb_disk_area = __dadd_rd(lb_area1, __dmul_rd(0.5, lb_area2));
	return lb_disk_area >= __dmul_ru(critical_ratio, ub_area_used);
}

static __device__ bool handle_case_2(IV la, IV r1, IV r2, IV r3, IV a13) {
	double ub_lane_container_area_ratio = __dadd_ru(1.0, -__dmul_rd(la.get_lb(), la.get_lb()));
	double lb_area1 = __dmul_rd(lb_pi, __dmul_rd(r1.get_lb(), r1.get_lb()));
	double lb_area2 = __dmul_rd(lb_pi, __dmul_rd(r2.get_lb(), r2.get_lb()));
	double lb_area3 = __dmul_rd(lb_pi, __dmul_rd(r3.get_lb(), r3.get_lb()));
	IV a = semicircle_angle_outer(r1) + a13;
	double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, a.get_ub()));
	double lb_disk_area = __dadd_rd(__dadd_rd(lb_area1, lb_area2), __dmul_rd(0.5, lb_area3));
	return lb_disk_area >= __dmul_ru(critical_ratio, ub_area_used);
}

static __global__ void kernel_find_critical_intervals_start_configuration(IV outer_la, Atomic_ordered_double *first_problematic_la, bool output_criticals) {
	for(int la_offset = blockIdx.x; la_offset < la_subintervals; la_offset += gridDim.x) {
		IV la = get_subinterval(outer_la, la_offset, la_subintervals);
		IV r1_range = compute_r1_range(la);
		for(int r1_offset = blockIdx.y; r1_offset < r1_subintervals; r1_offset += gridDim.y) {
			IV r1_ = get_subinterval(r1_range, r1_offset, r1_subintervals);
			IV r2_range = compute_r2_range(la, r1_);
			for(int r2_offset = threadIdx.x; r2_offset < r2_subintervals; r2_offset += blockDim.x) {
				IV r2_ = get_subinterval(r2_range, r2_offset, r2_subintervals);
				IV r3_range = compute_r3_range(la, r1_, r2_);
				for(int r3_offset = threadIdx.y; r3_offset < r3_subintervals; r3_offset += blockDim.y) {
					IV r3 = get_subinterval(r3_range, r3_offset, r3_subintervals);
					IV r2 = r2_;
					r2.tighten_lb(r3.get_lb());
					IV r1 = r1_;
					r1.tighten_lb(r2.get_lb());

					if(__dmul_ru(2.0, r1.get_ub()) >= __dadd_rd(1.0, -la.get_ub())) {
						printf("Start configuration: Circle possibly does not fit - la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g], r3 = [%.19g,%.19g]\n",
							la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub(), r3.get_lb(), r3.get_ub()
						);
					}

					IV a12 = collision_angle_outer_inner(la, r1, r2);
					IV a13 = collision_angle_outer_outer(r1, r3);
					IV a23 = collision_angle_inner_outer(la, r2, r3);

					UB case1 = (a12 + a23 > a13);
					if(possibly(case1) && !handle_case_1(la, r1, r2, a12)) {
						if(!output_criticals) {
							first_problematic_la->store_min(la.get_lb());
							return;
						} else {
							printf("Critical interval: Start configuration, case 1: la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g], r3 = [%.19g,%.19g]\n",
								la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub(), r3.get_lb(), r3.get_ub()
							);
						}
					}

					if(possibly(!case1) && !handle_case_2(la, r1, r2, r3, a13)) {
						if(!output_criticals) {
							first_problematic_la->store_min(la.get_lb());
							return;
						} else {
							printf("Critical interval: Start configuration, case 2: la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g], r3 = [%.19g,%.19g]\n",
								la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub(), r3.get_lb(), r3.get_ub()
							);
						}
					}
				}
			}
		}
	}
}

static __device__ bool handle_2d_case(IV la, IV r1, IV r2) {
	IV a1 = semicircle_angle_outer(r1);
	IV a2 = semicircle_angle_inner(la, r2);
	IV a12 = collision_angle_outer_inner(la, r1, r2);

	IV total_angle = a1 + a2 + a12;
	IV reuse_angle = a12 + a2 - a1;
	reuse_angle.tighten_lb(0.0);

	double ub_lane_container_area_ratio = __dadd_ru(1.0, -__dmul_rd(la.get_lb(), la.get_lb()));
	double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, total_angle.get_ub()));
	IV rho = 1.0 - (la + 2.0*r2).square();
	rho.tighten_lb(0.0);
	double lb_alpha_reusable = reuse_angle.get_lb();
	double lb_area_reusable = __dmul_rd(__dmul_rd(0.5, lb_alpha_reusable), rho.get_lb());
	double lb_area1 = __dmul_rd(lb_pi, __dmul_rd(r1.get_lb(), r1.get_lb()));
	double lb_area2 = __dmul_rd(lb_pi, __dmul_rd(r2.get_lb(), r2.get_lb()));
	double lb_disk_area = __dadd_rd(lb_area2, lb_area1);
	return lb_disk_area >= __dmul_ru(critical_ratio, __dadd_ru(ub_area_used, -lb_area_reusable));
}

static __global__ void kernel_find_critical_intervals_start_end_configuration_2d(IV outer_la, Atomic_ordered_double* first_problematic_la, bool output_criticals) {
	for(int la_offset = blockIdx.x; la_offset < la_subintervals_2d; la_offset += gridDim.x) {
		IV la = get_subinterval(outer_la, la_offset, la_subintervals_2d);
		IV r1_range = compute_r1_range(la);
		for(int r1_offset = blockIdx.y; r1_offset < r1_subintervals_2d; r1_offset += gridDim.y) {
			IV r1_ = get_subinterval(r1_range, r1_offset, r1_subintervals_2d);
			IV r2_range = compute_r2_range_2d(la, r1_);
			for(int r2_offset = threadIdx.x; r2_offset < r2_subintervals_2d; r2_offset += blockDim.x) {
				IV r2 = get_subinterval(r2_range, r2_offset, r2_subintervals_2d);
				IV r1 = r1_;
				r1.tighten_lb(r2.get_lb());

				if(!handle_2d_case(la, r1, r2)) {
					if(!output_criticals) {
						first_problematic_la->store_min(la.get_lb());
						return;
					} else {
						printf("Critical interval: Start & end configuration, 2 disks: la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g]\n",
							la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub()
						);
					}
				}
			}
		}
	}
}

void circlepacking::lane_packing::find_critical_intervals_start_configuration() {
	std::cout << "Start configuration..." << std::endl;
	run_kernel_until_no_criticals(&kernel_find_critical_intervals_start_configuration, IV(lambda_min,lambda_max), dim3(la_parallel, r1_parallel), dim3(r2_parallel, r3_parallel));
}

static void find_critical_intervals_start_end_configuration_2d() {
	std::cout << "Start & end configuration, 2 disks... " << std::endl;
	run_kernel_until_no_criticals(&kernel_find_critical_intervals_start_end_configuration_2d, IV(lambda_min, lambda_max), dim3(la_parallel_2d, r1_parallel_2d), dim3(r2_parallel_2d));
}

static bool __device__ handle_3d_case(IV la, IV r1, IV r2, IV r3) {
	IV a1 = semicircle_angle_outer(r1);
	IV a2 = semicircle_angle_inner(la, r2);
	IV a3 = semicircle_angle_outer(r3);
	IV a12 = collision_angle_outer_inner(la, r1, r2);
	IV a13 = collision_angle_outer_outer(r1, r2);
	IV a23 = collision_angle_inner_outer(la, r2, r3);
	
	// angular distance between center points of r1 and r3
	IV b13_123 = a12+a23;
	IV b13{a13.get_lb() < b13_123.get_lb() ? b13_123.get_lb() : a13.get_lb(), a13.get_ub() < b13_123.get_ub() ? b13_123.get_ub() : a13.get_ub()};

	IV a123 = a1 + b13 + a3;
	IV a122 = a1 + a12 + a2;
	IV a{a123.get_lb() < a122.get_lb() ? a122.get_lb() : a123.get_lb(), a123.get_ub() < a122.get_ub() ? a122.get_ub() : a123.get_ub()};

	double lb_reusable_angle = (a - a122).get_lb();
	if(lb_reusable_angle < 0.0) {
		lb_reusable_angle = 0.0;
	}
	double lb_reusable_angle_outer_rad = __dadd_rd(1.0, -__dmul_ru(2.0, r3.get_ub()));
	double lb_reusable_angle_outer_rad_sq = __dmul_rd(lb_reusable_angle_outer_rad, lb_reusable_angle_outer_rad);
	double ub_reusable_angle_inner_rad = la.get_ub();
	double ub_reusable_angle_inner_rad_sq = __dmul_ru(ub_reusable_angle_inner_rad, ub_reusable_angle_inner_rad);
	double lb_area_reused = __dmul_rd(__dmul_rd(0.5, lb_reusable_angle), __dadd_rd(lb_reusable_angle_outer_rad_sq, -ub_reusable_angle_inner_rad_sq));
	if(lb_area_reused < 0.0) {
		lb_area_reused = 0.0;
	}

	double ub_lane_container_area_ratio = __dadd_ru(1.0, -__dmul_rd(la.get_lb(), la.get_lb()));
	double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, a.get_ub()));
	ub_area_used = __dadd_ru(ub_area_used, -lb_area_reused);
	double lb_area1 = __dmul_rd(lb_pi, __dmul_rd(r1.get_lb(), r1.get_lb()));
	double lb_area2 = __dmul_rd(lb_pi, __dmul_rd(r2.get_lb(), r2.get_lb()));
	double lb_area3 = __dmul_rd(lb_pi, __dmul_rd(r3.get_lb(), r3.get_lb()));
	double lb_disk_area = __dadd_rd(lb_area1, __dadd_rd(lb_area2, lb_area3));

	return lb_disk_area >= __dmul_ru(critical_ratio, ub_area_used);
}

static void __global__ kernel_find_critical_intervals_start_end_configuration_3d(IV outer_la, Atomic_ordered_double* first_problematic_la, bool output_criticals) {
	for(int la_offset = blockIdx.x; la_offset < la_subintervals; la_offset += gridDim.x) {
		IV la = get_subinterval(outer_la, la_offset, la_subintervals);
		IV r1_range = compute_r1_range(la);
		for(int r1_offset = blockIdx.y; r1_offset < r1_subintervals; r1_offset += gridDim.y) {
			IV r1_ = get_subinterval(r1_range, r1_offset, r1_subintervals);
			IV r2_range = compute_r2_range(la, r1_);
			for(int r2_offset = threadIdx.x; r2_offset < r2_subintervals; r2_offset += blockDim.x) {
				IV r2_ = get_subinterval(r2_range, r2_offset, r2_subintervals);
				IV r3_range = compute_r3_range_3d(la, r1_, r2_);
				for(int r3_offset = threadIdx.y; r3_offset < r3_subintervals; r3_offset += blockDim.y) {
					IV r3 = get_subinterval(r3_range, r3_offset, r3_subintervals);
					IV r2 = r2_;
					r2.tighten_lb(r3.get_lb());
					IV r1 = r1_;
					r1.tighten_lb(r2.get_lb());

					if(!handle_3d_case(la, r1, r2, r3)) {
						if(!output_criticals) {
							first_problematic_la->store_min(la.get_lb());
							return;
						} else {
							printf("Start & end configuration: Critical interval - la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g], r3 = [%.19g,%.19g]\n",
								la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub(), r3.get_lb(), r3.get_ub()
							);
						}
					}
				}
			}
		}
	}
}

static void find_critical_intervals_start_end_configuration_3d() {
	std::cerr << "Start & end configuration, 3 disks ..." << std::endl;
	run_kernel_until_no_criticals(&kernel_find_critical_intervals_start_end_configuration_3d, IV(lambda_min, lambda_max), dim3(la_parallel, r1_parallel), dim3(r2_parallel, r3_parallel));
}

void circlepacking::lane_packing::find_critical_intervals_start_end_configuration() {
	find_critical_intervals_start_end_configuration_2d();
	find_critical_intervals_start_end_configuration_3d();
}

