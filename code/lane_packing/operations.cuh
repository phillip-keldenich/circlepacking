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

#ifndef CIRCLEPACKING_LANE_PACKING_OPERATIONS_CUH_INCLUDED_
#define CIRCLEPACKING_LANE_PACKING_OPERATIONS_CUH_INCLUDED_

#include "values.hpp"
#include <algcuda/interval.cuh>

namespace circlepacking {
	using IV = algcuda::Interval<double>;
	using UB = algcuda::Uncertain<bool>;

	namespace lane_packing {
		// compute a lower bound b on r (not r squared) such that all disks with radius
		// r >= b can be efficiently placed in a lane of height 1-lambda without considering
		// any other disks (provided that they fit into the lane at all)
		__device__ double lower_bound_radius_single_placement(double lb_lambda);

		// check whether the triangle given by its three sides hypot, s1, s2 is acute or obtuse
		__device__ UB is_obtuse(IV hypot, IV s1, IV s2);
		__device__ UB is_acute(IV hypot, IV s1, IV s2);

		// compute the collision angle between the rays induced by the _centers_
		// of two disks ra and rb; ra must be the larger disk; the disks may be
		// positioned touching the outer or the inner border of the lane
		__device__ IV collision_angle_inner_inner(IV la, IV ra, IV rb);
		__device__ IV collision_angle_outer_inner(IV la, IV ra, IV rb);
		__device__ IV collision_angle_inner_outer(IV la, IV ra, IV rb);
		__device__ IV collision_angle_outer_outer(IV ra, IV rb);

		// compute the angle used by a semicircle touching the outer border
		__device__ IV semicircle_angle_outer(IV r);
		__device__ IV semicircle_angle_inner(IV la, IV r);
	}
}

#endif

