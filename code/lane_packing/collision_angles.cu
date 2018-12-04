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
#include <algcuda/exit.cuh>

using namespace circlepacking;
using namespace circlepacking::lane_packing;

__device__ UB circlepacking::lane_packing::is_obtuse(IV hypot, IV s1, IV s2) {
	return hypot.square() > s1.square() + s2.square();
}

__device__ UB circlepacking::lane_packing::is_acute(IV hypot, IV s1, IV s2) {
	return hypot.square() < s1.square() + s2.square();
}

static inline __device__ IV compute_collision_angle(IV longer_to_center, IV shorter_to_center, IV ab) {
	longer_to_center.tighten_lb(shorter_to_center.get_lb());
	shorter_to_center.tighten_ub(longer_to_center.get_ub());

	IV result{0.0,-1.0};

	UB obt = is_obtuse(longer_to_center, shorter_to_center, ab);
	IV lc_sq = longer_to_center.square();
	IV sc_sq = shorter_to_center.square();

	IV add_term = (lc_sq-sc_sq) / ab;
	add_term.tighten_lb(0.0);

	if(possibly(obt)) {
		// handle the obtuse case
		IV lx = 0.5 * (add_term + ab);
		IV sx = 0.5 * (add_term - ab);
		sx.tighten_lb(0.0);
		sx.tighten_ub(shorter_to_center.get_ub());
		lx.tighten_ub(longer_to_center.get_ub());

		if(sx.empty() || lx.empty()) {
			if(definitely(obt)) {
				printf("Error: sx [%.19g,%.19g] or lx [%.19g,%.19g] is empty, but we are definitely in the obtuse case!\n", sx.get_lb(), sx.get_ub(), lx.get_lb(), lx.get_ub());
				algcuda::trap();
			}
		} else {
			// math says that there is a right angle at x; so we know
			// that lx cannot be the hypothenuse (even if intervals may disagree)
			IV lquot = lx/longer_to_center;
			lquot.tighten_ub(1.0);
			IV squot = sx/shorter_to_center;
			squot.tighten_ub(1.0);

			if(lquot.empty() || squot.empty()) {
				if(definitely(obt)) {
					printf("Error: squot [%.19g,%.19g] or lquot [%.19g,%.19g] is empty, but we are definitely in the obtuse case!\n", squot.get_lb(), squot.get_ub(), lquot.get_lb(), lquot.get_ub());
					algcuda::trap();
				}
			} else {
				IV alpha = asin(lquot);
				IV beta  = asin(squot);
				result = alpha-beta;
				result.tighten_lb(0.0);
			}
		}
	}

	if(!definitely(obt)) {
		// handle the acute case
		IV lx = 0.5 * (ab + add_term);
		IV sx = 0.5 * (ab - add_term);
		sx.tighten_lb(0.0);

		if(sx.empty()) {
			if(!possibly(obt)) {
				printf("Error: sx [%.19g,%.19g] is empty - shorter to center: [%.19g,%.19g], longer to center: [%.19g,%.19g], ab: [%.19g,%.19g], add_term: [%.19g,%.19g], but we are definitely in the acute case!\n",
						sx.get_lb(), sx.get_ub(),
						shorter_to_center.get_lb(), shorter_to_center.get_ub(),
						longer_to_center.get_lb(), longer_to_center.get_ub(),
						ab.get_lb(), ab.get_ub(),
						add_term.get_lb(), add_term.get_ub()
				);
				algcuda::trap();
			}
		} else {
			IV alpha = asin(lx/longer_to_center);
			IV beta  = asin(sx/shorter_to_center);
			beta.tighten_lb(0.0);
			IV r = alpha+beta;

			if(result.empty()) {
				result = r;
			} else {
				result.do_join(r);
			}
		}
	}

	return result;
}

__device__ IV circlepacking::lane_packing::collision_angle_inner_inner(IV la, IV ra, IV rb) {
	// this is monotonically increasing in ra, rb and monotonically decreasing in la; make use of this to improve accuracy
	double lb, ub;

	{ // compute upper bound
		IV longer_to_center  = la.get_lb() + IV{ra.get_ub(),ra.get_ub()};
		IV shorter_to_center = la.get_lb() + IV{rb.get_ub(),rb.get_ub()};
		IV ab                = IV{ra.get_ub(),ra.get_ub()} + rb.get_ub();
		ub = compute_collision_angle(longer_to_center, shorter_to_center, ab).get_ub();
	}

	{ // compute lower bound
		IV longer_to_center  = la.get_ub() + IV{ra.get_lb(),ra.get_lb()};
		IV shorter_to_center = la.get_ub() + IV{rb.get_lb(),rb.get_lb()};
		IV ab                = IV{ra.get_lb(),ra.get_lb()} + rb.get_lb();
		if(ab.get_lb() <= 0.0) {
			lb = 0.0;
		} else {
			lb = compute_collision_angle(longer_to_center, shorter_to_center, ab).get_lb();
		}
	}

	return {lb,ub};
}

static __device__ IV do_collision_angle_outer_inner(IV la, IV ra, IV rb) {
	IV ca = 1.0 - ra;
	IV cb = la + rb;
	IV ab = ra+rb;
	
	UB ca_longer = (cb <= ca);
	if(definitely(ca_longer)) {
		return compute_collision_angle(ca, cb, ab);
	} else if(definitely(!ca_longer)) {
		return compute_collision_angle(cb, ca, ab);
	} else {
		IV result = compute_collision_angle(ca, cb, ab);
		result.do_join(compute_collision_angle(cb, ca, ab));
		return result;
	}
}

__device__ IV circlepacking::lane_packing::collision_angle_outer_inner(IV la, IV ra, IV rb) {
	// this is monotonically increasing in ra, rb and not monotonic in la
	double lb, ub;
	
	if(possibly(2.0*(ra+rb) <= 1.0-la)) {
		lb = 0.0;
	} else {
		lb = do_collision_angle_outer_inner(la, IV(ra.get_lb(),ra.get_lb()), IV(rb.get_lb(),rb.get_lb())).get_lb();
	}

	ub = do_collision_angle_outer_inner(la, IV(ra.get_ub(),ra.get_ub()), IV(rb.get_ub(),rb.get_ub())).get_ub();

	return {lb,ub};
}

static __device__ IV do_collision_angle_inner_outer(IV la, IV ra, IV rb) {
	IV ca = la + ra;
	IV cb = 1.0 - rb;
	IV ab = ra+rb;
	
	UB ca_longer = (cb <= ca);
	if(definitely(ca_longer)) {
		return compute_collision_angle(ca, cb, ab);
	} else if(definitely(!ca_longer)) {
		return compute_collision_angle(cb, ca, ab);
	} else {
		IV result = compute_collision_angle(ca, cb, ab);
		result.do_join(compute_collision_angle(cb, ca, ab));
		return result;
	}
}

__device__ IV circlepacking::lane_packing::collision_angle_inner_outer(IV la, IV ra, IV rb) {
	// this is monotonically increasing in ra, rb and not monotonic in la
	double lb, ub;
	
	if(possibly(2.0*(ra+rb) <= 1.0-la)) {
		lb = 0.0;
	} else {
		lb = do_collision_angle_inner_outer(la, IV(ra.get_lb(),ra.get_lb()), IV(rb.get_lb(),rb.get_lb())).get_lb();
	}

	ub = do_collision_angle_inner_outer(la, IV(ra.get_ub(),ra.get_ub()), IV(rb.get_ub(),rb.get_ub())).get_ub();
	return {lb,ub};
}

__device__ IV circlepacking::lane_packing::collision_angle_outer_outer(IV ra, IV rb) {
	// this is monotonically increasing in ra, rb
	double lb, ub;

	{ // compute upper bound
		IV longer_to_center  = IV{1.0,1.0} - rb.get_ub();
		IV shorter_to_center = IV{1.0,1.0} - ra.get_ub();
		IV ab                = IV{ra.get_ub(),ra.get_ub()} + rb.get_ub();
		ub = compute_collision_angle(longer_to_center, shorter_to_center, ab).get_ub();
	}

	{ // compute lower bound
		IV longer_to_center  = IV{1.0,1.0} - rb.get_lb();
		IV shorter_to_center = IV{1.0,1.0} - ra.get_lb();
		IV ab                = IV{ra.get_lb(),ra.get_lb()} + rb.get_lb();
		
		if(ab.get_lb() <= 0.0) {
			lb = 0.0;
		} else {
			lb = compute_collision_angle(longer_to_center, shorter_to_center, ab).get_lb();
		}
	}

	return {lb,ub};
}

__device__ IV circlepacking::lane_packing::semicircle_angle_outer(IV r) {
	return {
		algcuda::detail::dasin_rd(__ddiv_rd(r.get_lb(), __dadd_ru(1.0,-r.get_lb()))),
		algcuda::detail::dasin_ru(__ddiv_ru(r.get_ub(), __dadd_rd(1.0,-r.get_ub())))
	};
}

__device__ IV circlepacking::lane_packing::semicircle_angle_inner(IV la, IV r) {
	// the function is monotonically increasing in r and decreasing in la
	return {
		algcuda::detail::dasin_rd(__ddiv_rd(r.get_lb(), __dadd_ru(la.get_ub(), r.get_lb()))),
		algcuda::detail::dasin_ru(__ddiv_ru(r.get_ub(), __dadd_rd(la.get_lb(), r.get_ub())))
	};
}

