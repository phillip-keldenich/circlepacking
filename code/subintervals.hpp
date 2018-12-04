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

#ifndef SUBINTERVALS_HPP_INCLUDED_
#define SUBINTERVALS_HPP_INCLUDED_

#include <algcuda/interval.hpp>

namespace circlepacking {
	// get the ith (almost) equal-width subinterval of n subintervals of the given interval; a full cover of the entire interval is guaranteed
	// intervals [0,n) are valid
	template<typename NumType>
	inline algcuda::Interval<NumType> __device__ __host__ get_subinterval(algcuda::Interval<NumType> interval, int i, int n) {
		double lb = interval.get_lb();
		double ub = interval.get_ub();	
		double diff = ub - lb;
		diff /= n;
		
		double l = lb + diff * i;
		double u = lb + diff * (i+1);
		if(i+1 == n) u = ub;
		return {l,u};
	}
}

#endif

