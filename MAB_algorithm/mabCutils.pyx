# distutils: language = c++
from libcpp.pair cimport pair


cdef extern from "src/cutils.h" namespace "mabCutils":
    cdef cppclass mabarraycpp:
        mabarraycpp() except +
        void startup(const int&)
        int size()
        void append(double)
        double avg()
        double& operator[](const int&)

    pair[double, int] getcatoni(double, int, double, mabarraycpp, double)

    double heavytail_pdf(double, double, double, double, double)


cdef class mabarray:
    cdef mabarraycpp wrapped

    def __cinit__(self, int maxsize):
        self.wrapped.startup(maxsize)

    def add(self, double v):
        self.wrapped.append(v)

    def avg(self):
        return self.wrapped.avg()

    def __len__(self):
        return self.wrapped.size()

    def __getitem__(self, int index):
        return self.wrapped[index]

    def __setitem__(self, int key, double val):
        self.wrapped[key] = val


def getCatoniMean(double v, int a, double z, mabarray mm, double tt):
    return getcatoni(v, a, z, mm.wrapped, tt)

def heavytail_dist_pdf(double alpha, double beta, double coef, double maxmomentorder, double x):
    return heavytail_pdf(alpha, beta, coef, maxmomentorder, x)
