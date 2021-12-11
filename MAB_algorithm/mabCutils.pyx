# distutils: language = c++

from libcpp.pair cimport pair

# extern

cdef extern from "src/cutils.h" namespace "mabCutils":
    cdef cppclass mabarraycpp:
        mabarraycpp() except +
        void startup(const int&)
        int size()
        void append(const double&)
        double avg()
        double& operator[](const int&)

    pair[double, int] getcatoni(const double &, const int &, double &, mabarraycpp &, const double &)

    double truncatedMean(const double &, const double &, const int &, mabarraycpp &)

    double meadianMean(const double &, const double &, const int &, mabarraycpp &)

    double heavytail_pdf(const double &, const double &, const double &, const double &, double)

# export: c class wrappers

cdef class mabarray:
    cdef mabarraycpp wrapped

    def __cinit__(self, const int &maxsize):
        self.wrapped.startup(maxsize)

    def add(self, const double &v):
        self.wrapped.append(v)

    def avg(self):
        return self.wrapped.avg()

    def __len__(self):
        return self.wrapped.size()

    def __getitem__(self, const int &index):
        return self.wrapped[index]

    def __setitem__(self, const int &key, const double &val):
        self.wrapped[key] = val

# export: c function wrappers

def getCatoniMean(const double &v, const int &a, double &z, mabarray arr, const double &tt):
    return getcatoni(v, a, z, arr.wrapped, tt)

def getTruncatedMean(const double &u, const double &ve, const int &itercount, mabarray arr):
    return truncatedMean(u, ve, itercount, arr.wrapped)

def getMedianMean(const double &v, const double &ve, const int &itercount, mabarray arr):
    return meadianMean(v, ve, itercount, arr.wrapped)

def heavytail_dist_pdf(const double &alpha, const double &beta, const double &coef, const double &maxmomentorder, double x):
    return heavytail_pdf(alpha, beta, coef, maxmomentorder, x)
