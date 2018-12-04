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

#ifndef ALGCUDA_INTERVAL_CUH_INCLUDED_
#define ALGCUDA_INTERVAL_CUH_INCLUDED_

#include <algcuda/interval.hpp>
#include <algcuda/exit.cuh>

namespace algcuda {
	template<> inline  __device__ Interval<float>& Interval<float>::operator+=(const Interval& o) noexcept {
		float l = __fadd_rd(lb, o.lb);
		float u = __fadd_ru(ub, o.ub);
		lb = l;
		ub = u;
		return *this;
	}

	template<> inline __device__ Interval<double>& Interval<double>::operator+=(const Interval& o) noexcept {
		double l = __dadd_rd(lb, o.lb);
		double u = __dadd_ru(ub, o.ub);
		lb = l;
		ub = u;
		return *this;
	}

	template<typename NumType> inline __device__ Interval<NumType>& Interval<NumType>::operator+=(NumType n) noexcept {
		return *this += Interval<NumType>{n,n};
	}

	template<typename NumType> inline __device__ Interval<NumType> operator+(const Interval<NumType>& a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a);
		result += b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator+(const Interval<NumType>& a, NumType b) noexcept {
		Interval<NumType> result(a);
		result += b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator+(NumType a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(b);
		result += a;
		return result;
	}

	template<> inline __device__ Interval<float>& Interval<float>::operator-=(const Interval& o) noexcept {
		float l = __fadd_rd(lb, -o.ub);
		float u = __fadd_ru(ub, -o.lb);
		lb = l;
		ub = u;
		return *this;
	}

	template<> inline __device__ Interval<double>& Interval<double>::operator-=(const Interval& o) noexcept {
		double l = __dadd_rd(lb, -o.ub);
		double u = __dadd_ru(ub, -o.lb);
		lb = l;
		ub = u;
		return *this;
	}

	template<typename NumType> inline __device__ Interval<NumType>& Interval<NumType>::operator-=(NumType n) noexcept {
		return *this -= Interval<NumType>{n,n};
	}

	template<typename NumType> inline __device__ Interval<NumType> operator-(const Interval<NumType>& a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a);
		result -= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator-(const Interval<NumType>& a, NumType b) noexcept {
		Interval<NumType> result(a);
		result -= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator-(NumType a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a,a);
		result -= b;
		return result;
	}

	template<> inline float __device__ Interval<float>::mul_rd(float a, float b) noexcept {
		return __fmul_rd(a,b);
	}

	template<> inline float __device__ Interval<float>::mul_ru(float a, float b) noexcept {
		return __fmul_ru(a,b);
	}

	template<> inline double __device__ Interval<double>::mul_rd(double a, double b) noexcept {
		return __dmul_rd(a,b);
	}

	template<> inline double __device__ Interval<double>::mul_ru(double a, double b) noexcept {
		return __dmul_ru(a,b);
	}

#ifdef ALGCUDA_UTILS_INTERVAL_USE_BRANCHING_MUL
	template<typename NumType> inline __device__ Interval<NumType>& Interval<NumType>::operator*=(const Interval& o) noexcept {
		NumType l, u;

		if(lb >= 0) {
			if(o.lb >= 0) {
				l = mul_rd(lb, o.lb);
				u = mul_ru(ub, o.ub);
			} else if(o.ub <= 0) {
				l = mul_rd(ub, o.lb);
				u = mul_ru(lb, o.ub);
			} else {
				l = mul_rd(ub, o.lb);
				u = mul_ru(ub, o.ub);
			}
		} else if(ub <= 0) {
			if(o.lb >= 0) {
				l = mul_rd(lb, o.ub);
				u = mul_ru(ub, o.lb);
			} else if(o.ub <= 0) {
				l = mul_rd(ub, o.ub);
				u = mul_ru(lb, o.lb);
			} else {
				l = mul_rd(lb, o.ub);
				u = mul_rd(lb, o.lb);
			}
		} else {
			if(o.lb >= 0) {
				l = mul_rd(lb, o.ub);
				u = mul_ru(ub, o.ub);
			} else if(o.ub <= 0) {
				l = mul_rd(ub, o.lb);
				u = mul_rd(lb, o.lb);
			} else {
				NumType l1 = mul_rd(lb, o.ub);
				NumType l2 = mul_rd(ub, o.lb);
				NumType u1 = mul_ru(lb, o.lb);
				NumType u2 = mul_ru(ub, o.ub);

				l = l1 < l2 ? l1 : l2;
				u = u1 > u2 ? u1 : u2;
			}
		}

		lb = l;
		ub = u;
		return *this;
	}
#else
	template<typename NumType> inline __device__ Interval<NumType>& Interval<NumType>::operator*=(const Interval& o) noexcept {
		double l1 = mul_rd(lb, o.lb), l2 = mul_rd(lb, o.ub), l3 = mul_rd(ub, o.lb), l4 = mul_rd(ub, o.ub);
		double u1 = mul_ru(lb, o.lb), u2 = mul_ru(lb, o.ub), u3 = mul_ru(ub, o.lb), u4 = mul_ru(ub, o.ub);
		
		double ll1 = l1 < l2 ? l1 : l2;
		double ll2 = l3 < l4 ? l3 : l4;
		double uu1 = u1 > u2 ? u1 : u2;
		double uu2 = u3 > u4 ? u3 : u4;
		lb = ll1 < ll2 ? ll1 : ll2;
		ub = uu1 > uu2 ? uu1 : uu2;
		return *this;
	}
#endif

	template<typename NumType> inline Interval<NumType>& __device__ Interval<NumType>::operator*=(NumType n) noexcept {
		if(n >= 0) {
			lb = mul_rd(lb, n);
			ub = mul_rd(ub, n);
		} else {
			NumType l = mul_rd(ub, n);
			NumType u = mul_ru(lb, n);
			lb = l;
			ub = u;
		}

		return *this;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator*(const Interval<NumType>& a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a);
		result *= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator*(const Interval<NumType>& a, NumType b) noexcept {
		Interval<NumType> result(a);
		result *= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator*(NumType a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(b);
		result *= a;
		return result;
	}

	template<> inline Interval<float> __device__ Interval<float>::reciprocal() const noexcept {
		if(lb > 0) {
			return {__frcp_rd(ub), __frcp_ru(lb)};
		} else if(ub < 0) {
			return {__frcp_rd(lb), __frcp_ru(ub)};
		} else {
			return {-(FLT_MAX * FLT_MAX), FLT_MAX * FLT_MAX};
		}
	}

	template<> inline Interval<double> __device__ Interval<double>::reciprocal() const noexcept {
		if(lb > 0) {
			return {__drcp_rd(ub), __drcp_ru(lb)};
		} else if(ub < 0) {
			return {__drcp_rd(lb), __drcp_ru(ub)};
		} else {
			return {-(DBL_MAX * DBL_MAX), DBL_MAX * DBL_MAX};
		}
	}

	template<> inline __device__ float Interval<float>::infty() noexcept {
		return FLT_MAX * FLT_MAX;
	}

	template<> inline __device__ double Interval<double>::infty() noexcept {
		return DBL_MAX * DBL_MAX;
	}

	template<> inline __device__ float Interval<float>::div_rd(float a, float b) noexcept {
		return __fdiv_rd(a,b);
	}

	template<> inline __device__ double Interval<double>::div_rd(double a, double b) noexcept {
		return __ddiv_rd(a,b);
	}

	template<> inline __device__ float Interval<float>::div_ru(float a, float b) noexcept {
		return __fdiv_ru(a,b);
	}

	template<> inline __device__ double Interval<double>::div_ru(double a, double b) noexcept {
		return __ddiv_ru(a,b);
	}

	template<typename NumType> inline Interval<NumType>& __device__ Interval<NumType>::operator/=(NumType nt) noexcept {
		NumType l,u;

		if(nt > 0) {
			l = div_rd(lb, nt);
			u = div_ru(ub, nt);
		} else if(nt <= 0) {
			l = div_rd(ub, nt);
			u = div_ru(lb, nt);
		}

		lb = l;
		ub = u;
		return *this;
	}

	template<typename NumType> inline Interval<NumType>& __device__ Interval<NumType>::operator/=(const Interval<NumType>& o) noexcept {
		NumType l, u;
		
		if(o.lb > 0) {
			if(lb >= 0) {
				l = div_rd(lb, o.ub);
				u = div_ru(ub, o.lb);
			} else if(ub <= 0) {
				l = div_rd(lb, o.lb);
				u = div_ru(ub, o.ub);
			} else {
				l = div_rd(lb, o.lb);
				u = div_ru(ub, o.lb);
			}
		} else if(o.ub < 0) {
			if(lb >= 0) {
				l = div_rd(ub, o.ub);
				u = div_ru(lb, o.lb);
			} else if(ub <= 0) {
				l = div_rd(ub, o.lb);
				u = div_ru(lb, o.ub);
			} else {
				l = div_rd(ub, o.ub);
				u = div_ru(lb, o.ub);
			}
		} else {
			u = infty();
			l = -u;
		}

		lb = l;
		ub = u;
		return *this;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator/(const Interval<NumType>& a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a);
		result /= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator/(const Interval<NumType>& a, NumType b) noexcept {
		Interval<NumType> result(a);
		result /= b;
		return result;
	}

	template<typename NumType> inline __device__ Interval<NumType> operator/(NumType a, const Interval<NumType>& b) noexcept {
		Interval<NumType> result(a, a);
		result /= b;
		return result;
	}

	template<> inline __device__ Interval<float> Interval<float>::sqrt() const noexcept {
		if(lb < 0.0) {
			printf("Interval with possibly negative value passed to Interval<float>::sqrt(): [%.19g,%.19g]\n", (double)lb, (double)ub);
			trap();
		}

		return { __fsqrt_rd(lb), __fsqrt_ru(ub) };
	}

	template<> inline __device__ Interval<double> Interval<double>::sqrt() const noexcept {
		if(lb < 0.0) {
			printf("Interval with possibly negative value passed to Interval<double>::sqrt(): [%.19g,%.19g]\n", lb, ub);
			trap();
		}

		return { __dsqrt_rd(lb), __dsqrt_ru(ub) };
	}

	template<typename NumType> Interval<NumType> __device__ sqrt(const Interval<NumType>& interval) {
		return interval.sqrt();
	}

	template<typename NumType> Interval<NumType> __device__ Interval<NumType>::square() const noexcept {
		Interval a = abs();
		return {__dmul_rd(a.get_lb(), a.get_lb()), __dmul_ru(a.get_ub(), a.get_ub())};
	}

	namespace detail {
		static const double lb_pi_two = 1.5707963267948966;
		static const double ub_pi_two = 1.5707963267948968;
		static const double lb_pi = 3.141592653589793;
		static const double ub_pi = 3.1415926535897936;
		static const double lb_3pi_two = 4.71238898038469;
		static const double ub_3pi_two = 4.712388980384691;
		static const double lb_two_pi = 6.283185307179586;
		static const double ub_two_pi = 6.283185307179587;

		// CUDA's library functions guarantee that the result of sin/cos is at most 1 ULP away from a
		// correctly rounded result which can again be 0.5ULPs away from the real result
		// because the documentation is not clear on how exactly this is meant,
		// especially regarding the case when the (exact or correctly rounded result) has
		// a larger exponent than the computed result, this means that we may have to go up to 3 steps from the computed result
		inline __device__ double sub_ulps_sc_range(double y) {
			y = nextafter(nextafter(nextafter(y,-2.0), -2.0), -2.0);
			if(y < -1.0) {
				y = -1.0;
			}
			return y;
		}

		inline __device__ double add_ulps_sc_range(double y) {
			y = nextafter(nextafter(nextafter(y,2.0), 2.0), 2.0);
			if(y > 1.0) {
				y = 1.0;
			}
			return y;
		}

		inline __device__ double dsin_rd(double x) {
			return sub_ulps_sc_range(::sin(x));
		}

		inline __device__ double dsin_ru(double x) {
			return add_ulps_sc_range(::sin(x));
		}
	
		inline __device__ double dcos_rd(double x) {
			return sub_ulps_sc_range(::cos(x));
		}

		inline __device__ double dcos_ru(double x) {
			return add_ulps_sc_range(::cos(x));
		}

		// acos is also at most 1 ULP from a correctly rounded result;
		inline __device__ double dacos_ru(double x) {
			double y = ::acos(x);
			y = nextafter(nextafter(nextafter(y, 4.0), 4.0), 4.0);
			if(y > ub_pi) { y = ub_pi; }
			return y;
		}

		inline __device__ double dacos_rd(double x) {
			double y = ::acos(x);
			y = nextafter(nextafter(nextafter(y, -1.0), -1.0), -1.0);
			if(y < 0.0) { y = 0.0; }
			return y;
		}

		// unlike acos, asin is at most 2 ULP from a correctly rounded result
		inline __device__ double dasin_rd(double x) {
			double y = ::asin(x);
			for(int i = 0; i < 5; ++i) {
				y = nextafter(y, -2.0);
			}
			if(y < -ub_pi_two) { y = -ub_pi_two; }
			if(x >= 0.0 && y < 0.0) { y = 0.0; }
			return y;
		}

		inline __device__ double dasin_ru(double x) {
			double y = ::asin(x);
			for(int i = 0; i < 5; ++i) {
				y = nextafter(y, 2.0);
			}
			if(y > ub_pi_two) { y = ub_pi_two; }
			if(x <= 0.0 && y > 0.0) { y = 0.0; }
			return y;
		}

		inline __device__ Interval<double> sin_in_range(Interval<double> x) {
			if(x.get_ub() <= ub_pi_two) {
				if(x.get_ub() >= lb_pi_two) {
					double ub = 1.0;
					double lb1 = dsin_rd(x.get_lb());
					double lb2 = dsin_rd(ub_pi_two);
					double lb = lb1 < lb2 ? lb1 : lb2;
					return {lb,ub};
				} else {
					return {dsin_rd(x.get_lb()), dsin_ru(x.get_ub())};
				}
			}

			if(x.get_ub() <= ub_3pi_two) {
				if(x.get_lb() >= ub_pi_two) {
					double ub = dsin_ru(x.get_lb());
					double lb = x.get_ub() >= lb_3pi_two ? -1.0 : dsin_rd(x.get_ub());
					return {lb,ub};
				} else {
					double lb1 = dsin_rd(x.get_lb());
					double lb2 = dsin_rd(x.get_ub());
					return {lb1 < lb2 ? lb1 : lb2 , 1.0};
				}
			}

			if(x.get_lb() <= ub_pi_two) {
				return {-1.0,1.0};
			}

			if(x.get_lb() <= ub_3pi_two) {
				double lb = -1.0;
				double ub1 = dsin_ru(x.get_lb());
				double ub2 = dsin_ru(x.get_ub());
				return {lb, ub1 > ub2 ? ub1 : ub2};
			} else {
				double lb = dsin_rd(x.get_lb());
				double ub = dsin_ru(x.get_ub());
				return {lb,ub};
			}
		}

		inline __device__ Interval<double> cos_in_range(Interval<double> x) {
			if(x.get_ub() <= ub_pi) {
				double lb = (x.get_ub() >= lb_pi ? -1.0 : dcos_rd(x.get_ub()));
				double ub = dcos_ru(x.get_lb());
				return {lb,ub};
			}

			double lb;
			double ub;
			if(x.get_lb() <= ub_pi) {
				lb = -1.0;
				double ub1 = dcos_ru(x.get_lb());
				double ub2 = dcos_ru(x.get_ub());
				ub = ub1 > ub2 ? ub1 : ub2;
			} else {
				lb = dcos_rd(x.get_lb());
				ub = dcos_ru(x.get_ub());
			}

			return {lb,ub};
		}

		inline __device__ Interval<double> cos_non_negative(Interval<double> x) {
			// if the value already is in range, shortcut
			if(x.get_lb() >= 0.0 && x.get_ub() <= ub_two_pi) {
				return cos_in_range(x);
			}

			// otherwise, normalize the value into the range [0, 2*pi]
			double lb_mult_2pi = __ddiv_rd(x.get_lb(), ub_two_pi);
			double ub_mult_2pi = __ddiv_ru(x.get_ub(), lb_two_pi);
			double lb_integral_part, ub_integral_part;
			modf(lb_mult_2pi, &lb_integral_part);
			modf(ub_mult_2pi, &ub_integral_part);

			double diff = ub_integral_part - lb_integral_part; 
			if(diff > 1.0) {
				return {-1.0,1.0};
			}
			
			if(diff == 0.0) {
				// we know that both ends of the interval are within one multiple of 2*pi
				double lb_range = __dadd_rd(x.get_lb(), -__dmul_ru(lb_integral_part, ub_two_pi));
				double ub_range = __dadd_ru(x.get_ub(), -__dmul_rd(lb_integral_part, lb_two_pi));
				if(lb_range < 0.0) { lb_range = 0.0; }
				if(lb_range > ub_two_pi) { lb_range = ub_two_pi; }
				if(ub_range < 0.0) { ub_range = 0.0; }
				if(ub_range > ub_two_pi) { ub_range = ub_two_pi; }
				return cos_in_range(Interval<double>(lb_range, ub_range));
			}

			// otherwise, the ends of the interval are in two adjacent multiples of 2*pi
			double lb_range = __dadd_rd(x.get_lb(), -__dmul_ru(lb_integral_part, ub_two_pi));
			if(lb_range < 0.0) { return {-1.0,1.0}; }
			if(lb_range > ub_two_pi) { lb_range = ub_two_pi; }
			
			double ub_range = __dadd_ru(x.get_ub(), -__dmul_rd(ub_integral_part, lb_two_pi));
			if(ub_range > ub_two_pi) { return {-1.0,1.0}; }
			if(ub_range < 0.0) { ub_range = 0.0; }
			return cos_in_range(Interval<double>(lb_range, ub_two_pi)).join(cos_in_range(Interval<double>(0.0, ub_range)));
		}

		inline __device__ Interval<double> sin_non_negative(Interval<double> x) {
			// if the value already is in range, shortcut
			if(x.get_lb() >= 0.0 && x.get_ub() <= ub_two_pi) {
				return sin_in_range(x);
			}

			// otherwise, normalize the value into the range [0, 2*pi]
			double lb_mult_2pi = __ddiv_rd(x.get_lb(), ub_two_pi);
			double ub_mult_2pi = __ddiv_ru(x.get_ub(), lb_two_pi);
			double lb_integral_part, ub_integral_part;
			modf(lb_mult_2pi, &lb_integral_part);
			modf(ub_mult_2pi, &ub_integral_part);

			double diff = ub_integral_part - lb_integral_part; 
			if(diff > 1.0) {
				return {-1.0,1.0};
			}
			
			if(diff == 0.0) {
				// we know that both ends of the interval are within one multiple of 2*pi
				double lb_range = __dadd_rd(x.get_lb(), -__dmul_ru(lb_integral_part, ub_two_pi));
				double ub_range = __dadd_ru(x.get_ub(), -__dmul_rd(lb_integral_part, lb_two_pi));
				if(lb_range < 0.0) { lb_range = 0.0; }
				if(lb_range > ub_two_pi) { lb_range = ub_two_pi; }
				if(ub_range < 0.0) { ub_range = 0.0; }
				if(ub_range > ub_two_pi) { ub_range = ub_two_pi; }
				return sin_in_range(Interval<double>(lb_range, ub_range));
			}

			// otherwise, the ends of the interval are in two adjacent multiples of 2*pi
			double lb_range = __dadd_rd(x.get_lb(), -__dmul_ru(lb_integral_part, ub_two_pi));
			if(lb_range < 0.0) { return {-1.0,1.0}; }
			if(lb_range > ub_two_pi) { lb_range = ub_two_pi; }
			
			double ub_range = __dadd_ru(x.get_ub(), -__dmul_rd(ub_integral_part, lb_two_pi));
			if(ub_range > ub_two_pi) { return {-1.0,1.0}; }
			if(ub_range < 0.0) { ub_range = 0.0; }
			return sin_in_range(Interval<double>(lb_range, ub_two_pi)).join(sin_in_range(Interval<double>(0.0, ub_range)));
		}
	}
	
	__device__ Interval<double> sin(Interval<double> x) {
		if(x.get_lb() >= 0.0) {
			return detail::sin_non_negative(x);
		} else if(x.get_ub() <= 0.0) {
			return -detail::sin_non_negative(-x);
		} else {
			Interval<double> xneg{0.0, -x.get_lb()};
			Interval<double> xpos{0.0, x.get_ub()};
			return detail::sin_non_negative(xpos).join(-detail::sin_non_negative(xneg));
		}
	}

	__device__ Interval<double> cos(Interval<double> x) {
		if(x.get_lb() >= 0.0) {
			return detail::cos_non_negative(x);
		} else if(x.get_ub() <= 0.0) {
			return detail::cos_non_negative(-x);
		} else {
			double neg_lb = -x.get_lb();
			Interval<double> range(0.0, neg_lb < x.get_ub() ? x.get_ub() : neg_lb);
			return detail::cos_non_negative(range);
		}
	}

	__device__ Interval<double> acos(Interval<double> x) {
		if(x.get_lb() < -1.0 || x.get_ub() > 1.0) {
			printf("Interval with value possibly out of range given to acos(Interval<double>): [%.19g,%.19g]!\n", x.get_lb(), x.get_ub());
			trap();
		}
	
		return {detail::dacos_rd(x.get_ub()), detail::dacos_ru(x.get_lb())};
	}

	__device__ Interval<double> asin(Interval<double> x) {
		if(x.get_lb() < -1.0 || x.get_ub() > 1.0) {
			printf("Interval with value possibly out of range given to asin(Interval<double>): [%.19g,%.19g]!\n", x.get_lb(), x.get_ub());
			trap();
		}

		return {detail::dasin_rd(x.get_lb()), detail::dasin_ru(x.get_ub())};
	}
}

#endif

