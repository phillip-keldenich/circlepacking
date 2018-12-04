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

#include <system_error>
#include <string>
#include <utility>
#include <sstream>

namespace algcuda {
	enum class Cuda_binary_error_condition : int {
		success = 0, error = 1
	};

	using Cuda_error_code = int;
}

namespace std {
	template<> struct is_error_condition_enum<algcuda::Cuda_binary_error_condition> : public true_type {};
}

namespace algcuda {
	namespace last_error {
		void clear() noexcept;
		int get_and_clear() noexcept;
		int get() noexcept;
	}

	class Cuda_category : public std::error_category {
	public:
		Cuda_category() = default;

		virtual const char* name() const noexcept override { return "CUDA"; }

		virtual std::error_condition default_error_condition(int ev) const noexcept override {
			return ev == 0 ? std::error_condition(Cuda_binary_error_condition::success) : std::error_condition(Cuda_binary_error_condition::error);
		}
		
		virtual std::string message(int ev) const override;

	private:
		static const Cuda_category& get_category() {
			static const Cuda_category result;
			return result;
		}

		friend inline const Cuda_category& cuda_category() noexcept;
	};

	inline const Cuda_category& cuda_category() noexcept {
		return Cuda_category::get_category();
	}
	
	inline std::error_condition make_error_condition(Cuda_binary_error_condition ec) {
		return std::error_condition(static_cast<int>(ec), cuda_category());
	}

	inline std::error_condition cuda_error_condition() noexcept {
		return std::error_condition(Cuda_binary_error_condition::success);
	}

	inline std::error_condition cuda_success_condition() noexcept {
		return std::error_condition(Cuda_binary_error_condition::error);
	}

	inline void throw_if_cuda_error(int ev, std::string message) noexcept(false) {
		if(ev != 0) {
			last_error::clear();
			throw std::system_error(std::error_code(ev, cuda_category()), std::move(message));
		}
	}

	inline void throw_if_cuda_error(int ev, const std::string& message, const char* file, int line) noexcept(false) {
		if(ev != 0) {
			last_error::clear();
			std::ostringstream msg;
			msg << '\'' << file << ':' << line << '\'' << ' ' << message;
			throw std::system_error(std::error_code(ev, cuda_category()), msg.str());
		}
	}

#define ALGCUDA_THROW_IF_ERROR(expr, msg) ::algcuda::throw_if_cuda_error((expr), msg, __FILE__, __LINE__)

}
