#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <lawrap/blas.h> 

namespace py = pybind11;


py::array_t<double> J_build(py::array_t<double> g,
                            py::array_t<double> D)
{
    py::buffer_info g_info = g.request();
    py::buffer_info D_info = D.request();


    if(g_info.ndim != 4)
        throw std::runtime_error("g is not four dimensional array");

    if(D_info.ndim != 2)
        throw std::runtime_error("D is not a two dimensional array");

    if(g_info.shape[2]*g_info.shape[3] != D_info.shape[0]*D_info.shape[1])
        throw std::runtime_error("r*s of g != r*s of D");

    size_t nbf = g_info.shape[0];

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



py::array_t<double> K_build(py::array_t<double> g,
                            py::array_t<double> D)
{
    py::buffer_info g_info = g.request();
    py::buffer_info D_info = D.request();


    if(g_info.ndim != 4)
        throw std::runtime_error("g is not four dimensional array");

    if(D_info.ndim != 2)
        throw std::runtime_error("D is not a two dimensional array");

    if(g_info.shape[2]*g_info.shape[3] != D_info.shape[0]*D_info.shape[1])
        throw std::runtime_error("r*s of g != r*s of D");

    size_t nbf = g_info.shape[0];

    const double * g_data = static_cast<double *>(g_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);

    std::vector<double> K_data(nbf * nbf);

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






//py::array_t<double> dgemm_numpy(double alpha,
//                                py::array_t<double> A,
//                                py::array_t<double> B)
//{
//    py::buffer_info A_info = A.request();
//    py::buffer_info B_info = B.request();
//
//    if(A_info.ndim != 2)
//        throw std::runtime_error("A is not a matrix");
//
//    if(B_info.ndim != 2)
//        throw std::runtime_error("B is not a matrix");
//
//    if(A_info.shape[1] != B_info.shape[0])
//        throw std::runtime_error("Rows of A != Columns of B");
//    
//
//    size_t C_nrows = A_info.shape[0];
//    size_t C_ncols = B_info.shape[1];
//    size_t n_k = A_info.shape[1]; // Same as B_info.shape[0]
//
//    const double * A_data = static_cast<double *>(A_info.ptr);
//    const double * B_data = static_cast<double *>(B_info.ptr);
//
//    std::vector<double> C_data(C_nrows * C_ncols);
//
//
//    for(size_t i = 0; i < C_nrows; i++)
//    {
//        for(size_t j = 0; j < C_ncols; j++)
//        {
//            double val = 0.0;
//            for(size_t k = 0; k < n_k; k++)
//            {
//                val += A_data[i * n_k + k] * B_data[k * C_ncols + j];
//            }
//
//            val *= alpha;
//
//            C_data[i * C_ncols + j] = val;
//        }
//    }
//
//
//    py::buffer_info C_buf = 
//    {
//        C_data.data(),
//        sizeof(double),
//        py::format_descriptor<double>::format(),
//        2,
//        { C_nrows, C_ncols },
//        { C_ncols * sizeof(double), sizeof(double) }
//    };
//
//    return py::array_t<double>(C_buf);
//}



PYBIND11_PLUGIN(jk_build)
{
    py::module m("jk_build", "qm2 jk competition module");

    m.def("J_build", &J_build, "Builds the J Matrix from g and D");
    m.def("K_build", &K_build, "Builds the K Matrix from g and D");
    //m.def("dgemm_numpy", &dgemm_numpy, "Prints the degemm of two matrices using numpy");

    return m.ptr();
}

