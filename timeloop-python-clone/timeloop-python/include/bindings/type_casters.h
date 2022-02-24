#pragma once

// Timeloop headers
#include "workload/util/per-data-space.hpp"

// PyBind11 headers
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

// Type casters
namespace pybind11 {
namespace detail {
template <typename Type>
struct type_caster<problem::PerDataSpace<Type>> {
  using value_conv = make_caster<Type>;

  bool load(handle src, bool convert) {
    if (!isinstance<sequence>(src) || isinstance<bytes>(src) ||
        isinstance<str>(src)) {
      return false;
    }
    auto l = reinterpret_borrow<sequence>(src);
    if (l.size() != problem::GetShape()->NumDataSpaces) {
      return false;
    }
    size_t ctr = 0;
    for (auto it : l) {
      value_conv conv;
      if (!conv.load(it, convert)) return false;
      value[ctr++] = cast_op<Type &&>(std::move(conv));
    }
    return true;
  }

  template <typename T>
  static handle cast(T &&src, return_value_policy policy, handle parent) {
    list l(src.size());
    size_t index = 0;
    for (auto &&value : src) {
      auto value_ = reinterpret_steal<object>(
          value_conv::cast(forward_like<T>(value), policy, parent));
      if (!value_) return handle();
      PyList_SET_ITEM(l.ptr(), (ssize_t)index++,
                      value_.release().ptr());  // steals a reference
    }
    return l.release();
  }

  PYBIND11_TYPE_CASTER(problem::PerDataSpace<Type>, _("PerDataSpace"));
};
}  // namespace detail
}  // namespace pybind11
