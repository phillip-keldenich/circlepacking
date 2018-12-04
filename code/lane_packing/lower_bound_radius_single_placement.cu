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
#include <algcuda/interval.cuh>

__device__ double circlepacking::lane_packing::lower_bound_radius_single_placement(double lb_lambda) {
	static const double lb_pi = 3.141592653589793;
	const double ub_rhs_factor = __ddiv_ru(critical_ratio, lb_pi);
	const double ub_rhs = __dmul_ru(ub_rhs_factor, __dadd_ru(1.0, -__dmul_rd(lb_lambda,lb_lambda)));

	// we cannot have any disks with radius > 0.25; using that as an upper bound
	// makes sure that we cannot return an upper bound that is too small, even if
	// it does not work for any r
	// r^2/arcsin(x/1-x) is monotonically increasing in [0,0.39]; therefore, we can actually
	// return an upper bound
	double ub = 0.25;
	double lb = 5.0e-324;

	// perform bisection to find the smallest value of r for which single placement definitely works
	for(;;) {
		double mid = 0.5*(lb+ub);
		if(mid <= lb || mid >= ub) {
			return ub;
		}

		double r_sq = __dmul_rd(mid, mid);
		double ub_quot = __ddiv_ru(mid, __dadd_rd(1.0, -mid));

		// asin is monotonically increasing, so we do not actually need an interval
		double ub_asin = algcuda::detail::dasin_ru(ub_quot);
		double lb_lhs = __ddiv_rd(r_sq, ub_asin);
			
		if(lb_lhs >= ub_rhs) {
			// this radius definitely works
			ub = mid;
		} else {
			// this radius potentially does not work
			lb = mid;
		}
	}
}

