# distutils: language = c++
import numpy
cimport numpy
from libcpp.pair cimport pair

cdef extern from "src/cutils.h" namespace "mabCutils":
    cdef cppclass node:
        node() except +
        node(double) except +
        node* next
        double val

    cdef cppclass mabnodescpp:
        mabnodescpp() except +
        mabnodescpp(double) except +
        mabnodescpp(node*) except +
        mabnodescpp(node) except +
        mabnodescpp(mabnodescpp) except +
        int size
        node* next
        node* tail
        node* gethead()
        node* gettail()
        int size()
        void append(double)
        double avg()

    pair[double, int] getcatoni(double, int, double, mabnodescpp, double)

# def _sumup(numpy.ndarray[numpy.float64_t, ndim=1] arr):
#     for i,v in enumerate(arr):
#         arr[i] = f(v)
cdef class mabnodes:
    cdef mabnodescpp wrappednode

    def __cinit__(self):
        self.wrappednode = mabnodescpp()

    def add(self, v):
        self.wrappednode.append(v)

    def avg(self):
        return self.wrappednode.avg()

    def __len__(self):
        return self.wrappednode.size()

def getCatoniMean(double v, int a, double z, mabnodes mm, double tt):
    return getcatoni(v, a, z, mm.wrappednode, tt)
