# distutils: language = c++
# cython: cdivision=True


from libcpp.vector cimport vector
from libc.math cimport sqrt
from cython.parallel import prange

cdef int N_CPU = 4

cdef int A(int i, int j) nogil:
    return ((i+j) * (i+j+1) / 2 + i + 1)

cdef double dot(vector[double] & v, vector[double] & u, int n) nogil:
    cdef int i
    cdef double summ = 0
    for i in prange(n, nogil=True, num_threads=N_CPU):
        summ += v[i] * u[i]
    return summ


cdef void mult_Av_helper(vector[double] & v, vector[double] & out, int i, int n) nogil:
    cdef double summ = 0
    cdef int tmp
    for j in range(n):
        tmp = A(i, j)
        summ += v[j] / tmp
    out[i] = summ


cdef void mult_Atv_helper(vector[double] & v, vector[double] & out, int i, int n) nogil:
    cdef double summ = 0
    cdef int tmp
    for j in range(n):
        tmp = A(j, i)
        summ += v[j] / tmp
    out[i] = summ


cdef void mult_Av(vector[double] & v, vector[double] & out, int n):
    cdef int i, j
    for i in prange(n, nogil=True, num_threads=N_CPU):
        mult_Av_helper(v, out, i, n)

cdef void mult_Atv(vector[double] & v, vector[double] & out, int n):
    cdef int i, j
    cdef double summ
    cdef int tmp
    for i in prange(n, nogil=True, num_threads=N_CPU):
        mult_Atv_helper(v, out, i, n)

cdef void mult_AtAv(vector[double] & tmp, vector[double] & v, vector[double] & out, int n):
    mult_Av(v, tmp, n)
    mult_Atv(tmp, out, n)


cpdef main():
    cdef int n
    import sys
    n = int(sys.argv[1])
    cdef vector[double] u = vector[double]()
    cdef vector[double] v = vector[double]()
    cdef vector[double] tmp=vector[double]()
    cdef int i
    u.resize(n, 1)
    v.resize(n)
    tmp.resize(n)
    for i in range(10):
        mult_AtAv(tmp, u, v, n)
        mult_AtAv(tmp, v, u, n)
    print('%.9f' % sqrt(dot(u,v,n) / dot(v,v,n)))


