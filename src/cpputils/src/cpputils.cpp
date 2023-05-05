#include <iostream>

#include "fastbasis.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;


py::array_t<double> make_AS_wrapper(py::array_t<double> array_old, py::array_t<double> array_new) {
  // check input dimensions
  if ( array_old.ndim()     != 4 )
    throw std::runtime_error("Input should be 4-D NumPy array");
  if ( array_new.ndim()     != 4 )
    throw std::runtime_error("Input should be 4-D NumPy array");
  
  int N = array_old.shape()[0];
  for(int i = 1; i < 4; i++) {
    if( array_old.shape()[i] != N )
      throw std::runtime_error("All dimensions of old must be equal");
    if( array_new.shape()[i] != N )
      throw std::runtime_error("All dimensions of new must be equal");
  }

  for(int i = 0; i < 4; i++) {
    if( array_old.shape()[i] != array_new.shape()[i] )
      throw std::runtime_error("Differing array dims on old and new");
  }

  auto buf_old = array_old.request();
  double* ptr_old = (double*) buf_old.ptr;
 
  auto buf_new = array_new.request();
  double* ptr_new = (double*) buf_new.ptr;

  make_AS(ptr_old, ptr_new, N);

  return array_new;
}

PYBIND11_MODULE(cpputils, m) {
  m.doc() = "pybind11 plugin";
  // m.def("sum", [](std::vector<double> data) {
  //           sum(data.data(), data.size());
  //           return data;
  //       });
  m.def("make_AS", &make_AS_wrapper, "sdasd");
}
