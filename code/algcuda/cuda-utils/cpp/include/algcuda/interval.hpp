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

#ifndef ALGCUDA_INTERVAL_HPP_INCLUDED_
#define ALGCUDA_INTERVAL_HPP_INCLUDED_

#include <cfloat>
#include <iostream>
#include <iomanip>
#include <algcuda/macros.hpp>

namespace algcuda {
	template<typename Type> struct Uncertain {
		Uncertain() noexcept = default;
		Uncertain(const Uncertain&) noexcept = default;
		Uncertain &operator=(const Uncertain&) noexcept = default;

		explicit __host__ __device__ Uncertain(Type v) noexcept : lb(v), ub(v) {}
		__host__ __device__ Uncertain(Type l, Type u) noexcept : lb(l), ub(u) {}

		bool __host__ __device__ is_certain() const noexcept {
			return lb == ub;
		}

		bool __host__ __device__ is_uncertain() const noexcept {
			return lb != ub;
		}

		Type __host__ __device__ get_lb() const noexcept {
			return lb;
		}

		Type __host__ __device__ get_ub() const noexcept {
			return ub;
		}

		inline Uncertain __host__ __device__ operator!() const noexcept;

	private:
		Type lb, ub;
	};

	static inline __host__ __device__ bool definitely(Uncertain<bool> b) {
		return b.get_lb();
	}

	static inline __host__ __device__ bool possibly(Uncertain<bool> b) {
		return b.get_ub();
	}

	static inline __host__ __device__ bool definitely_not(Uncertain<bool> b) {
		return !b.get_ub();
	}

	static inline __host__ __device__ bool possibly_not(Uncertain<bool> b) {
		return !b.get_lb();
	}

	static_assert(std::is_pod<Uncertain<bool>>::value, "Uncertain<bool> must be POD!");

	template<> inline __host__ __device__ Uncertain<bool> Uncertain<bool>::operator!() const noexcept {
		return {!ub, !lb};
	}

	template<typename NumType> class Interval {
	public:
		static_assert(std::is_floating_point<NumType>::value, "Interval NumType must be floating-point type!");

		Interval() noexcept = default;
		Interval(const Interval&) noexcept = default;
		Interval& operator=(const Interval&) noexcept = default;

		explicit __host__ __device__ Interval(NumType v) noexcept :
			lb(v), ub(v)
		{}

		__host__ __device__ Interval(NumType l, NumType u) noexcept :
			lb(l), ub(u)
		{}

		NumType __host__ __device__ get_lb() const noexcept {
			return lb;
		}

		NumType __host__ __device__ get_ub() const noexcept {
			return ub;
		}

		void __host__ __device__ set_lb(NumType n) noexcept {
			lb = n;
		}

		void __host__ __device__ set_ub(NumType n) noexcept {
			ub = n;
		}

		bool __host__ __device__ contains(NumType n) const noexcept {
			return lb <= n && n <= ub;
		}

		inline Interval<NumType> reciprocal() const noexcept;
		inline __device__ Interval &operator+=(const Interval& o) noexcept;
		inline __device__ Interval &operator+=(NumType n) noexcept;
		inline __device__ Interval &operator-=(const Interval& o) noexcept;
		inline __device__ Interval &operator-=(NumType n) noexcept;
		inline __device__ Interval &operator*=(const Interval& o) noexcept;
		inline __device__ Interval &operator*=(NumType n) noexcept;
		inline __device__ Interval &operator/=(const Interval& o) noexcept;
		inline __device__ Interval &operator/=(NumType n) noexcept;
		inline __device__ Interval sqrt() const noexcept;
		Interval __device__  __host__ operator-() const noexcept { return {-ub, -lb}; }

		inline __device__ Interval square() const noexcept;

		Uncertain<bool> __device__ __host__ operator< (NumType n) const noexcept {
			return { ub < n, lb < n };
		}

		Uncertain<bool> __device__ __host__ operator> (NumType n) const noexcept {
			return { lb > n, ub > n };
		}

		Uncertain<bool> __device__ __host__ operator<=(NumType n) const noexcept {
			return { ub <= n, lb <= n };
		}

		Uncertain<bool> __device__ __host__ operator>=(NumType n) const noexcept {
			return { lb >= n, ub >= n };
		}

		Uncertain<bool> __device__ __host__ operator< (const Interval& o) const noexcept {
			return {
				ub < o.lb, // lb = true iff we are definitely less than o
				lb < o.ub  // ub = true iff we are possibly less than o
			};
		}

		Uncertain<bool> __device__ __host__ operator<=(const Interval& o) const noexcept {
			return {
				ub <= o.lb, // lb = true iff we are definitely less than or equal to o
				lb <= o.ub  // ub = true iff we are possibly   less than or equal to o
			};
		}

		Uncertain<bool> __device__ __host__ operator==(const Interval& o) const noexcept {
			return {lb == ub && o.lb == o.ub && lb == o.lb, !(lb > o.ub) && !(ub < o.lb) };
		}

		Uncertain<bool> __device__ __host__ operator==(NumType n) const noexcept {
			return { lb == ub && lb == n, !(lb > n) && !(ub < n) };
		}

		Interval __device__ __host__ intersect(const Interval& o) const noexcept {
			return Interval(lb < o.lb ? o.lb : lb, ub > o.ub ? o.ub : ub);
		}

		Interval& __device__ __host__ do_intersect(const Interval& o) noexcept {
			if(o.lb > lb)
				lb = o.lb;

			if(o.ub < ub)
				ub = o.ub;

			return *this;
		}

		Interval __device__ __host__ join(const Interval& o) const noexcept {
			return Interval(lb > o.lb ? o.lb : lb, ub < o.ub ? o.ub : ub);
		}

		Interval& __device__ __host__ do_join(const Interval& o) noexcept {
			if(o.lb < lb)
				lb = o.lb;

			if(o.ub > ub)
				ub = o.ub;

			return *this;
		}

		void __device__ __host__ tighten_ub(const NumType ub) noexcept {
			if(ub < this->ub) {
				this->ub = ub;
			}
		}

		void __device__ __host__ tighten_lb(const NumType lb) noexcept {
			if(this->lb < lb) {
				this->lb = lb;
			}
		}

		bool __device__ __host__ empty() const noexcept {
			return lb > ub;
		}

		__device__ __host__ Interval abs() const noexcept {
			NumType l{0}, u{0};

			if(ub > 0) {
				u = ub;
			} else {
				l = -ub;
			}

			if(lb < 0) {
				u = u < -lb ? -lb : u;
			} else {
				l = lb;
			}

			return {l, u};
		}

	private:
		NumType lb, ub;

		static inline __device__ NumType mul_rd(NumType n1, NumType n2) noexcept;
		static inline __device__ NumType mul_ru(NumType n1, NumType n2) noexcept;
		static inline __device__ NumType div_rd(NumType n1, NumType n2) noexcept;
		static inline __device__ NumType div_ru(NumType n1, NumType n2) noexcept;
		static inline __device__ NumType infty() noexcept;
	};

#ifdef __CUDACC__
	static inline __device__ Interval<double> sin(Interval<double> x);
	static inline __device__ Interval<double> cos(Interval<double> x);
	static inline __device__ Interval<double> asin(Interval<double> x);
	static inline __device__ Interval<double> acos(Interval<double> x);
#endif

	template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator==(NumType nt, const Interval<NumType>& i) noexcept {
		return i == nt;
	}

	template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator!=(const Interval<NumType>& i, NumType nt) noexcept {
		return !(i == nt);
	}

	template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator!=(NumType nt, const Interval<NumType>& i) noexcept {
		return i != nt;
	}

	template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator!=(const Interval<NumType>& i1, const Interval<NumType>& i2) noexcept {
		return !(i1 == i2);
	}

	template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator< (NumType nt, const Interval<NumType>& i) noexcept {
		return i > nt;
	}

	template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator> (NumType nt, const Interval<NumType>& i) noexcept {
		return i < nt;
	}

	template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator<=(NumType nt, const Interval<NumType>& i) noexcept {
		return i >= nt;
	}

	template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator>=(NumType nt, const Interval<NumType>& i) noexcept {
		return i <= nt;
	}

	template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator> (const Interval<NumType>& i1, const Interval<NumType>& i2) noexcept {
		return i2 < i1;
	}

	template<typename NumType> static inline __device__ __host__ Uncertain<bool> operator>=(const Interval<NumType>& i1, const Interval<NumType>& i2) noexcept {
		return i2 <= i1;
	}

	static_assert(std::is_pod<Interval<float>>::value,  "Float intervals must be POD!");
	static_assert(std::is_pod<Interval<double>>::value, "Double intervals must be POD!");

	template<typename NumType>
		static inline __host__ std::ostream& operator<<(std::ostream& output, const Interval<NumType>& iv)
	{
		std::ios_base::fmtflags f = output.flags(std::ios::right);
		std::streamsize p = output.precision(19);
		output << '[' << std::setw(26) << iv.get_lb() << ", " << std::setw(26) << iv.get_ub() << ']';
		output.flags(f);
		output.precision(p);
		return output;
	}

	template<typename CharType, typename NumType>
		static inline __host__ std::basic_istream<CharType>& operator>>(std::basic_istream<CharType>& input, Interval<NumType>& iv) {
			CharType c;
			if(!(input >> c)) {
				return input;
			}

			if(c == '[') {
				NumType n1;
				NumType n2;

				CharType comma;
				CharType end;

				if(!(input >> n1) || !(input >> comma)) {
					return input;
				}

				if(comma != ',' && comma != ';') {
					input.setstate(std::ios_base::failbit);
					return input;
				}

				if(!(input >> n2) || !(input >> end)) {
					return input;
				}

				if(end != ']') {
					input.setstate(std::ios_base::failbit);
					return input;
				}

				iv = Interval<NumType>{n1,n2};
			} else {
				if(!input.putback(c)) {
					return input;
				}
				
				NumType result;
				if(!(input >> result)) {
					return input;
				}
				iv = Interval<NumType>{result,result};
			}

			return input;
		}

	struct Interval_compare {
		template<typename NumType> __device__ __host__ bool operator()(const Interval<NumType>& i1, const Interval<NumType>& i2) const noexcept {
			return i1.get_lb() < i2.get_lb() || (i1.get_lb() == i2.get_lb() && i1.get_ub() < i2.get_ub());
		}
	};
}

namespace std {
	template<typename NumType> struct hash<algcuda::Interval<NumType>> {
		std::size_t __host__ __device__ operator()(const algcuda::Interval<NumType>& i) const noexcept {
			std::size_t h1 = std::hash<NumType>{}(i.get_lb());
			std::size_t h2 = std::hash<NumType>{}(i.get_ub());
			return h2 + static_cast<std::size_t>(0x9e3779b97f4a7c15ull) + (h1 << 6) + (h1 << 2);
		}
	};
}

#endif

