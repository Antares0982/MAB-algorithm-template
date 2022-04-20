# distutils: language = c++

from libcpp.pair cimport pair

# extern

cdef extern from "cutils.h" namespace "mabCutils":
    cdef cppclass mabarraycpp:
        mabarraycpp() except +
        void startup(int)
        int size()
        void append(double)
        double avg()
        double& operator[](int)

    cdef cppclass medianOfMeanArrayCpp:
        medianOfMeanArrayCpp() except +
        void startup(int)
        int size()
        void append(double)
        double avg()
        double& operator[](int)
        # extra interface
        double getMedianMean(int)

    cdef pair[double, int] getcatoni(const double, const int, double &, mabarraycpp &, const double)

    cdef double truncatedMean(const double, const double, const int, mabarraycpp &)

    cdef double medianMean(const double, const double, const int, mabarraycpp &)

    cdef double heavytail_pdf(const double, const double, const double, const double, double)

# export: c class wrappers

cdef class mabarray:
    cdef mabarraycpp wrapped

    def __cinit__(self, const int maxsize):
        self.wrapped.startup(maxsize)

    def add(self, const double v):
        self.wrapped.append(v)

    def avg(self):
        return self.wrapped.avg()

    def __len__(self):
        return self.wrapped.size()

    def __getitem__(self, const int index):
        return self.wrapped[index]

    def __setitem__(self, const int key, const double val):
        self.wrapped[key] = val

cdef class medianOfMeanArray:
    cdef medianOfMeanArrayCpp wrapped

    def __cinit__(self, const int maxsize):
        self.wrapped.startup(maxsize)

    def add(self, const double v):
        self.wrapped.append(v)

    def avg(self):
        return self.wrapped.avg()

    def __len__(self):
        return self.wrapped.size()

    def __getitem__(self, const int index):
        return self.wrapped[index]

    def __setitem__(self, const int key, const double val):
        self.wrapped[key] = val

    def medianMean(self, int binsizeN):
        return self.wrapped.getMedianMean(binsizeN)

# export: c function wrappers

def getCatoniMean(const double v, const int a, double z, mabarray arr, const double tt):
    return getcatoni(v, a, z, arr.wrapped, tt)

def getTruncatedMean(const double u, const double ve, const int itercount, mabarray arr):
    return truncatedMean(u, ve, itercount, arr.wrapped)

def getMedianMean(const double v, const double ve, const int itercount, mabarray arr):
    return medianMean(v, ve, itercount, arr.wrapped)

def heavytail_dist_pdf(const double alpha, const double beta, const double coef, const double maxmomentorder, double x):
    return heavytail_pdf(alpha, beta, coef, maxmomentorder, x)
