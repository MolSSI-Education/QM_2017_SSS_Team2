#include <lawrap/blas.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cassert>
#include <iostream>
namespace py = pybind11;

class JKBuilder {
  private:
  py::array_t<double> g_arr;
  // We will want these for the K_Build algo
  size_t nbf;

  public:
  // Default constructor
  //JKBuilder() = default;
  // Default destructor
  //~JKBuilder() = default;
  // Overload for the constructor we will use
  // initializer list grantees that g_buff is set before the rest happens
  JKBuilder(py::array_t<double> g_in)
      : g_arr(g_in)
  {
    py::buffer_info g_info = g_arr.request();

    // beter than if, throw because it will be a no-op when building in release mode
    // in debug mode it will actually do the check then throw
    assert(g_info.ndim == 4);
    assert(g_info.format == py::format_descriptor<double>::format());

    nbf = g_info.shape[3];
    // The strides are in bytes we need to convert to element to we divide by the size of an
    // elements
  }
  py::array_t<double> J_build(py::array_t<double> D_arr)
  {
    std::vector<double> J_vec(nbf * nbf);
    py::buffer_info g_info = g_arr.request();
    py::buffer_info D_info = D_arr.request();
    const double* g_data = static_cast<double*>(g_info.ptr);
    const double* D_data = static_cast<double*>(D_info.ptr);

    LAWrap::gemv(
        'T', nbf * nbf, nbf * nbf, 1.0, g_data, nbf * nbf, D_data, 1, 0.0, J_vec.data(), 1);
    return py::array_t<double>(
        py::buffer_info(J_vec.data(), sizeof(double), py::format_descriptor<double>::format(), 2,
            { nbf, nbf }, { nbf * sizeof(double), sizeof(double) }));
  }
  py::array_t<double> K_build(py::array_t<double> D_arr)
  {
    std::vector<double> K_vec(nbf * nbf);
    py::buffer_info g_info = g_arr.request();
    py::buffer_info D_info = D_arr.request();
    const double* g_data = static_cast<double*>(g_info.ptr);
    const double* D_data = static_cast<double*>(D_info.ptr);

    double* K_data = K_vec.data();

    const size_t g_p_stride = nbf * nbf * nbf;
    const size_t g_q_stride = nbf * nbf;
    const size_t g_r_stride = nbf;
    const size_t g_s_stride = 1;
    const size_t D_q_stride = nbf;
    const size_t D_s_stride = 1;
    const size_t K_r_stride = 1;
    const size_t K_p_stride = nbf;

    for (size_t np = 0; np < nbf; np++) {
      for (size_t nr = 0; nr < nbf; nr++) {
        for(size_t nq = 0; nq < nbf; nq++){

        // Dot product along s (unit strides)
        // Dereference K_data ptr to add to the element (not to the ptr)
            K_data[np*nbf + nr] += LAWrap::dot(nbf, &(g_data[nq*g_q_stride]),
                g_s_stride, &(D_data[nq*D_q_stride]), D_s_stride);

            //Post increment ptrs with r stride
        }
        g_data += g_r_stride;
      }
      // post increment ptrs with p stride
      g_data += g_p_stride - (g_r_stride*nbf); // need to "undo the r strides"
    }
    return py::array_t<double>(
        py::buffer_info(K_vec.data(), sizeof(double), py::format_descriptor<double>::format(), 2,
            { nbf, nbf }, { nbf * sizeof(double), sizeof(double) }));
  }
};

// J_ref = np.einsum("pqrs,rs->pq", g, D)
// K_ref = np.einsum("pqrs,qs->pr", g, D)

PYBIND11_PLUGIN(jk_build)
{
  py::module m("jk_build", "qm2 jk competition module");

  py::class_<JKBuilder>(m, "JKBuilder")
      .def(py::init<py::array_t<double> >())
      .def("build_J_only", &JKBuilder::J_build, "Builds J given (arg1) a density")
      .def("build_K_only", &JKBuilder::K_build, "Builds K given (arg1) a density")
      .def("compute", [](JKBuilder& jk, py::array_t<double> D) {
        py::array_t<double> J = jk.J_build(D);
        py::array_t<double> K = jk.K_build(D);
        return py::make_tuple(J, K);
      },"Builds both J and K returned as a tuple given (arg1) a density");

  return m.ptr();
}
