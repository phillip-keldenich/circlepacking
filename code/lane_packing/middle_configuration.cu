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
		void find_critical_intervals_middle_configuration();
	}
}

using namespace circlepacking;
using namespace circlepacking::lane_packing;
using algcuda::Atomic_ordered_double;

static constexpr int la_subintervals = 512;
static constexpr int r1_subintervals = 256;
static constexpr int r2_subintervals = 256;
static constexpr int r3_subintervals = 64;

static constexpr int la_parallel = 128;
static constexpr int r1_parallel = 64;
static constexpr int r2_parallel = 32;
static constexpr int r3_parallel = 16;

// middle configuration: we consider the disks r_1, r_2, r_3;
// r1 is not the first disk in the lane, and r_3 is not the last in the lane
// this means that r2+r3 must be >= 1/2 * lane_width; in particular,
// even r3 must be >= 1/4*lane_width

// cases:
// case 1: r_1 is on the inner side
//   case 1.1: r_2 is stopped by r_1, not a previous disk
//      case 1.1.1: r_3 is stopped by r_2
//         2-disk case; we check the segment between r_1 and r_2 only
//      case 1.1.2: r_3 is stopped by r_1
//         3-disk case: we check the segment between r_1 and r_3, taking all of r_2 into account
//   case 1.2: r_2 is stopped by r_0, a previous disk, not by r_1 => either r1 is not on the path (we need not consider it),
//         i.e., if r_3 is stopped by r2, or this case is the same as 1.1.2
//
// case 2: r_1 is on the outer side (analogously)

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

static inline __device__ IV compute_r3_range(IV la, IV r1, IV r2) {
	double lb_by_lane_width = (0.25*(1.0-la)).get_lb();
	return {lb_by_lane_width, r2.get_ub()};
}

static __device__ bool handle_case_1_1(IV la, IV r1, IV r2, IV r3) {
	// in this case, we assume that r1 stops r2
	IV collision_angle_r12 = collision_angle_inner_outer(la, r1, r2);
	IV collision_angle_r13 = collision_angle_inner_inner(la, r1, r3);
	IV collision_angle_r23 = collision_angle_outer_inner(la, r2, r3);

	double ub_lane_container_area_ratio = __dadd_ru(1.0, -__dmul_rd(la.get_lb(), la.get_lb()));
	UB case_111 = (collision_angle_r12 + collision_angle_r23 >= collision_angle_r13);
	double lb_area1 = __dmul_rd(lb_pi, __dmul_rd(r1.get_lb(), r1.get_lb()));
	double lb_area2 = __dmul_rd(lb_pi, __dmul_rd(r2.get_lb(), r2.get_lb()));

	if(possibly(case_111)) {
		// the collision angle of r1--r3 is at most the collision angle of r1--r2 + r2--r3,
		// i.e. r2 stops r3; here, we only consider the piece between r1 and r2
 		double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, collision_angle_r12.get_ub()));
		double lb_disk_area = __dmul_rd(0.5, __dadd_rd(lb_area1, lb_area2));
		if(lb_disk_area < __dmul_ru(critical_ratio, ub_area_used)) {
			return false;
		}
	}

	if(possibly(!case_111)) {
		// the collision angle of r1--r3 is greater than the collision angle of r1--r2 + r2--r3,
		// i.e. r1 stops r3; here, we only consider the piece between r1 and r3
 		double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, collision_angle_r13.get_ub()));
		double lb_area3     = __dmul_rd(lb_pi, __dmul_rd(r3.get_lb(), r3.get_lb()));
		double lb_disk_area = __dadd_rd(__dmul_rd(0.5, __dadd_rd(lb_area1, lb_area3)), lb_area2);
		if(lb_disk_area < __dmul_ru(critical_ratio, ub_area_used)) {
			return false;
		}
	}

	return true;
}

static __device__ bool handle_case_2_1(IV la, IV r1, IV r2, IV r3) {
	// in this case, we assume that r1 stops r2
	IV collision_angle_r12 = collision_angle_outer_inner(la, r1, r2);
	IV collision_angle_r13 = collision_angle_outer_outer(r1, r3);
	IV collision_angle_r23 = collision_angle_inner_outer(la, r2, r3);

	double ub_lane_container_area_ratio = __dadd_ru(1.0, -__dmul_rd(la.get_lb(), la.get_lb()));
	UB case_211 = (collision_angle_r12 + collision_angle_r23 >= collision_angle_r13);
	double lb_area1 = __dmul_rd(lb_pi, __dmul_rd(r1.get_lb(), r1.get_lb()));
	double lb_area2 = __dmul_rd(lb_pi, __dmul_rd(r2.get_lb(), r2.get_lb()));

	if(possibly(case_211)) {
		// the collision angle of r1--r3 is at most the collision angle of r1--r2 + r2--r3,
		// i.e. r2 stops r3; here, we only consider the piece between r1 and r2
		double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, collision_angle_r12.get_ub()));
		double lb_disk_area = __dmul_rd(0.5, __dadd_rd(lb_area1, lb_area2));
		if(lb_disk_area < __dmul_ru(critical_ratio, ub_area_used)) {
			return false;
		}
	}

	if(possibly(!case_211)) {
		// the collision angle of r1--r3 is greater than the collision angle of r1--r2 + r2--r3,
		// i.e. r1 stops r3; here, we only consider the piece between r1 and r3
		double ub_area_used = __dmul_ru(ub_lane_container_area_ratio, __dmul_ru(0.5, collision_angle_r13.get_ub()));
		double lb_area3     = __dmul_rd(lb_pi, __dmul_rd(r3.get_lb(), r3.get_lb()));
		double lb_disk_area = __dadd_rd(__dmul_rd(0.5, __dadd_rd(lb_area1, lb_area3)), lb_area2);
		if(lb_disk_area < __dmul_ru(critical_ratio, ub_area_used)) {
			return false;
		}
	}

	return true;
}

static __global__ void kernel_find_critical_intervals_middle_configuration(IV outer_la, Atomic_ordered_double *first_problematic_la, bool output_criticals) {
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
						printf("Middle configuration: Circle possibly does not fit - la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g], r3 = [%.19g,%.19g]\n",
							la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub(), r3.get_lb(), r3.get_ub()
						);
						algcuda::trap();
					}

					if(!handle_case_1_1(la, r1, r2, r3)) {
						if(output_criticals) {
							printf("Critical interval: Middle configuration, case 1.1: la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g], r3 = [%.19g,%.19g]\n",
								la.get_lb(), la.get_ub(), r1.get_lb(), r1.get_ub(), r2.get_lb(), r2.get_ub(), r3.get_lb(), r3.get_ub()
							);
						} else {
							first_problematic_la->store_min(la.get_lb());
							return;
						}
					}

					if(!handle_case_2_1(la, r1, r2, r3)) {
						if(output_criticals) {
							printf("Critical interval: Middle configuration, case 2.1: la = [%.19g,%.19g], r1 = [%.19g,%.19g], r2 = [%.19g,%.19g], r3 = [%.19g,%.19g]\n",
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

void circlepacking::lane_packing::find_critical_intervals_middle_configuration() {
	const dim3 grid(la_parallel, r1_parallel);
	const dim3 block(r2_parallel, r3_parallel);
	std::cout << "Middle configuration ..." << std::endl;
	run_kernel_until_no_criticals(&kernel_find_critical_intervals_middle_configuration, IV(lambda_min, lambda_max), grid, block);
}

