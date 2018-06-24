# distutils: language = c++
# cython: cdivision=True

from libcpp.vector cimport vector
from libc.math cimport sqrt
from cython.parallel import prange
cdef int A(int i, int j):
    return ((i+j) * (i+j+1) / 2 + i + 1)

cdef double dot(vector[double] & v, vector[double] & u, int n):
    cdef int i
    cdef double summ = 0
    for i in range(n):
        summ += v[i] * u[i]
    return summ


cdef void mult_Av(vector[double] & v, vector[double] & out, int n):
    cdef int i, j
    cdef double summ
    cdef int tmp
    for i in range(n):
        summ = 0
        for j in range(n):
            tmp = A(i, j)
            summ += v[j] / tmp
        out[i] = summ

cdef void mult_Atv(vector[double] & v, vector[double] & out, int n):
    cdef int i, j
    cdef double summ
    cdef int tmp
    for i in range(n):
        summ = 0
        for j in range(n):
            tmp = A(j, i)
            summ += v[j] / tmp
        out[i] = summ

cdef void mult_AtAv(vector[double] & v, vector[double] & out, int n):
    cdef vector[double] tmp=vector[double](n)
    mult_Av(v, tmp, n)
    mult_Atv(tmp, out, n)


cpdef main():
    import sys
    n = int(sys.argv[1])
    cdef vector[double] u = vector[double]()
    cdef vector[double] v = vector[double]()
    cdef int i
    u.resize(n)
    v.resize(n)
    for i in range(n):
        u[i] = 1
    for i in range(10):
        mult_AtAv(u, v, n)
        mult_AtAv(v, u, n)
    print('%.9f' % sqrt(dot(u,v,n) / dot(v,v,n)))


