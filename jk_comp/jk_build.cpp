#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <lawrap/blas.h> 

namespace py = pybind11;

class JKBuilder
{
    public:
        // Object Members
        py::array_t<double> g_J;
        py::array_t<double> g_K;

        // Functions
        py::array_t<double> J_build(py::array_t<double> D);
        py::array_t<double> K_build(py::array_t<double> D);

        JKBuilder(py::array_t<double>);
};


JKBuilder::JKBuilder (py::array_t<double> g): g_J(g)
{
    // Read in g[p][q][r][s] as g_J since it is already properly indexed for J_build
    py::buffer_info g_J_info = g_J.request();   
    if(g_J_info.ndim != 4)
        throw std::runtime_error("g_J is not four dimensional array");
    size_t nbf = g_J_info.shape[0];
    const double * g_J_data = static_cast<double *>(g_J_info.ptr);


    // Translate g[p][q][r][s] to g[p][r][q][s] for proper indexing for K_build
    std::vector<double> g_K_data(nbf*nbf*nbf*nbf);

    for(int p = 0; p < nbf; p++)
    {
        for(int r = 0; r < nbf; r++)
        {
            for(int q = 0; q < nbf; q++)
            {
                for(int s = 0; s < nbf; s++)
                {
                    int g_K_offset = p * (nbf*nbf*nbf) + r * (nbf*nbf) + q * (nbf) + s;
                    int g_J_offset = p * (nbf*nbf*nbf) + q * (nbf*nbf) + r * (nbf) + s;
                    
                    g_K_data[g_K_offset] = g_J_data[g_J_offset];
                }
            }
        }
    }

    py::buffer_info g_K_buf = 
    {
        g_K_data.data(),
        sizeof(double),
        py::format_descriptor<double>::format(),
        4,
        { nbf, nbf, nbf, nbf },
        { nbf * nbf * nbf * sizeof(double), nbf * nbf * sizeof(double), nbf * sizeof(double), sizeof(double) }
    };

    g_K = py::array_t<double>(g_K_buf);
}






py::array_t<double> JKBuilder::J_build(py::array_t<double> D)
{
    py::buffer_info g_J_info = g_J.request();
    py::buffer_info D_info = D.request();

    if(g_J_info.ndim != 4)
        throw std::runtime_error("g is not four dimensional array");

    if(D_info.ndim != 2)
        throw std::runtime_error("D is not a two dimensional array");

    if(g_J_info.shape[2]*g_J_info.shape[3] != D_info.shape[0]*D_info.shape[1])
        throw std::runtime_error("r*s of g != r*s of D");

    size_t nbf = g_J_info.shape[0];

    const double * g_J_data = static_cast<double *>(g_J_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);

    std::vector<double> J_data(nbf * nbf);

    //g[i][j] = g[i * row + j]
    //g[p][q][r][s] = g[pq * len(pq) + rs]
    //g[p][q][r][s] = g[(p * len(p) + q) * len(pq) + (r * len(rs) + s)]

    LAWrap::gemv('T',
            nbf*nbf,
            nbf*nbf,
            1.0,
            g_J_data,
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
    py::buffer_info g_K_info = g_K.request();
    py::buffer_info D_info = D.request();

    if(g_K_info.ndim != 4)
        throw std::runtime_error("g is not four dimensional array");

    if(D_info.ndim != 2)
        throw std::runtime_error("D is not a two dimensional array");

    if(g_K_info.shape[2]*g_K_info.shape[3] != D_info.shape[0]*D_info.shape[1])
        throw std::runtime_error("r*s of g != r*s of D");

    size_t nbf = g_K_info.shape[0];

    const double * g_K_data = static_cast<double *>(g_K_info.ptr);
    const double * D_data = static_cast<double *>(D_info.ptr);

    std::vector<double> K_data(nbf * nbf);

    //g[i][j] = g[i * row + j]
    //g[p][q][r][s] = g[pq * len(pq) + rs]
    //g[p][q][r][s] = g[(p * len(p) + q) * len(pq) + (r * len(rs) + s)]

    LAWrap::gemv('T',
            nbf*nbf,
            nbf*nbf,
            1.0,
            g_K_data,
            nbf*nbf,
            D_data,
            1,
            0.0,
            K_data.data(),
            1
            );

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
//py::array_t<double> JKBuilder::K_build(py::array_t<double> g,
//                                       py::array_t<double> D)
//{
//    py::buffer_info g_info = g.request();
//    py::buffer_info D_info = D.request();
//
//
//    if(g_info.ndim != 4)
//        throw std::runtime_error("g is not four dimensional array");
//
//    if(D_info.ndim != 2)
//        throw std::runtime_error("D is not a two dimensional array");
//
//    if(g_info.shape[2]*g_info.shape[3] != D_info.shape[0]*D_info.shape[1])
//        throw std::runtime_error("r*s of g != r*s of D");
//
//    size_t nbf = g_info.shape[0];
//
//    const double * g_data = static_cast<double *>(g_info.ptr);
//    const double * D_data = static_cast<double *>(D_info.ptr);
//
//    std::vector<double> K_data(nbf * nbf);
//
//    for(int p = 0; p < nbf; p++)
//    {
//        for(int r = 0; r < nbf; r++)
//        {   
//            K_data[p * nbf + r] = 0.0;
//            for(int q = 0; q < nbf; q++)
//            {
//                size_t g_offset = p*(nbf*nbf*nbf) + q*(nbf*nbf) + r*nbf;
//                size_t D_offset = q*nbf;
//                const double* g_ptr = &(g_data[g_offset]);
//                const double* D_ptr = &(D_data[D_offset]);
//                
//                // Dot Product over s
//                K_data[p * nbf + r] += LAWrap::dot(nbf, g_ptr, 1, D_ptr, 1);
//            }
//        }   
//    }
//
//
//    py::buffer_info K_buf = 
//    {
//        K_data.data(),
//        sizeof(double),
//        py::format_descriptor<double>::format(),
//        2,
//        { nbf, nbf },
//        { nbf * sizeof(double), sizeof(double) }
//    };
//
//    return py::array_t<double>(K_buf);
//}


//PYBIND11_PLUGIN(jk_build)
//{
//    py::module m("jk_build", "qm2 jk competition module");
//
//    m.def("J_build", &JKBuilder.J_build, "Builds the J Matrix from g and D");
//    //m.def("K_build", &K_build, "Builds the K Matrix from g and D");
//
//    return m.ptr();
//}


PYBIND11_PLUGIN(jk_build)
{
  py::module m("jk_build", "This line creates the module we are goign to export to python");

  // m.def("J_build", &J_build, "This will bind a c++ function called cool_function with the python side name some_neat_function");
 

  py::class_<JKBuilder>(m, "JKBuilder", "This will bind the c++ class AclassName to the module m (see above"
  				"Python side the class will be called BoundClass " 
			    "also did you know that these pieces will be treated as one string?!"
				"Note: no comas!")
    .def(py::init<py::array_t<double>>()) // this defines the constructor as a function that takes two arguments one is an int the other a double
    .def("J_build", &JKBuilder::J_build, "this binds the function member_function to the python verison of the class") //<-- note this semicolon everything from py::class.. to here was one c++ statment! 
    .def("K_build", &JKBuilder::K_build, "this binds the function member_function to the python verison of the class"); //<-- note this semicolon everything from py::class.. to here was one c++ statment! 

  return m.ptr();

}
