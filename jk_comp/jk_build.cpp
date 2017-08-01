#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <lawrap/blas.h> 
#include <omp.h>

namespace py = pybind11;

class JKBuilder
{
    public:
        // Object Members
        py::array_t<double> g;
        size_t nbf;

        // Functions
        py::array_t<double> J_build(py::array_t<double> D);
        py::array_t<double> K_build(py::array_t<double> D);

        JKBuilder(py::array_t<double>);
};


JKBuilder::JKBuilder (py::array_t<double> g_eri): g(g_eri)
{
    py::buffer_info g_info = g.request();   
    if(g_info.ndim != 4)
        throw std::runtime_error("g is not four dimensional array");
    nbf = g_info.shape[0];
}



py::array_t<double> JKBuilder::J_build(py::array_t<double> D)
{
    py::buffer_info g_info = g.request();   
    py::buffer_info D_info = D.request();

    if(D_info.ndim != 2)
        throw std::runtime_error("D is not a two dimensional array");

    if(g_info.shape[2]*g_info.shape[3] != D_info.shape[0]*D_info.shape[1])
        throw std::runtime_error("r*s of g != r*s of D");

    const double * g_data = static_cast<double *>(g_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);

    std::vector<double> J_data(nbf * nbf);

    //g[i][j] = g[i * row + j]
    //g[p][q][r][s] = g[pq * len(pq) + rs]
    //g[p][q][r][s] = g[(p * len(p) + q) * len(pq) + (r * len(rs) + s)]

    LAWrap::gemv('T',
            nbf*nbf,
            nbf*nbf,
            1.0,
            g_data,
            nbf*nbf,
            D_data,
            1,
            0.0,
            J_data.data(),
            1
            );

    py::buffer_info J_buf = 
    {
        J_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        { nbf, nbf },
        { nbf * sizeof(double), sizeof(double) }
    };

    return py::array_t<double>(J_buf);
}



py::array_t<double> JKBuilder::K_build(py::array_t<double> D)
{
    py::buffer_info g_info = g.request();   
    py::buffer_info D_info = D.request();

    if(D_info.ndim != 2)
        throw std::runtime_error("D is not a two dimensional array");

    if(g_info.shape[2]*g_info.shape[3] != D_info.shape[0]*D_info.shape[1])
        throw std::runtime_error("r*s of g != r*s of D");

    const double * g_data = static_cast<double *>(g_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);

    std::vector<double> K_data(nbf * nbf);

    //g[i][j] = g[i * row + j]
    //g[p][q][r][s] = g[pq * len(pq) + rs]
    //g[p][q][r][s] = g[(p * len(p) + q) * len(pq) + (r * len(rs) + s)]
    
#pragma omp parallel for schedule(dynamic)
    for(int p = 0; p < nbf; p++)
    {
        for(int r = 0; r < nbf; r++)
        {   
            K_data[p * nbf + r] = 0.0;
            for(int q = 0; q < nbf; q++)
            {
                size_t g_offset = p*(nbf*nbf*nbf) + q*(nbf*nbf) + r*nbf;
                size_t D_offset = q*nbf;
                const double* g_ptr = &(g_data[g_offset]);
                const double* D_ptr = &(D_data[D_offset]);
                
                // Dot Product over s
                K_data[p * nbf + r] += LAWrap::dot(nbf, g_ptr, 1, D_ptr, 1);
            }
        }   
    }

    py::buffer_info K_buf = 
    {
        K_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        2,
        { nbf, nbf },
        { nbf * sizeof(double), sizeof(double) }
    };

    return py::array_t<double>(K_buf);
}


PYBIND11_PLUGIN(jk_build)
{
  py::module m("jk_build", "This line creates the module we are going to export to python");

  py::class_<JKBuilder>(m, "JKBuilder", "This will bind the c++ class JKBuilder to the module m (see above)"
  				"Python side the class will be called JKBuilder ")
    .def(py::init<py::array_t<double>>()) // this defines the constructor as a function that takes one argument: an py::array_t<double>
    .def("J_build", &JKBuilder::J_build, "this binds the function J_build to the python verison of the class") 
    .def("K_build", &JKBuilder::K_build, "this binds the function K_build to the python verison of the class"); //<-- note this semicolon everything from py::class.. to here was one c++ statment! 

  return m.ptr();

}
