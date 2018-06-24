# distutils: language = c++
# cython: cdivision=True


from libcpp.vector cimport vector
from libc.math cimport sqrt
from cython.parallel import prange


cdef extern from "emmintrin.h":  # in this example, we use SSE2
    ctypedef double __m128d

    __m128d _mm_set_pd (double __A,double __B) nogil
    __m128d _mm_add_pd (__m128d __A, __m128d __B) nogil
    __m128d _mm_div_pd (__m128d __A, __m128d __B) nogil
    __m128d _mm_setzero_pd() nogil
    void _mm_store_pd (double *__P, __m128d __A) nogil

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
    cdef __m128d summ = _mm_setzero_pd()
    cdef int tmp
    cdef __m128d a, b
    cdef double[2] double_arr
    cdef int j
    for j in range(0, n, 2):
        b = _mm_set_pd(v[j], v[j+1])
        a = _mm_set_pd(A(i, j), A(i, j + 1))
        summ = _mm_add_pd(summ, _mm_div_pd(b, a))
    _mm_store_pd(double_arr, summ)
    out[i] = double_arr[0] + double_arr[1]


cdef void mult_Atv_helper(vector[double] & v, vector[double] & out, int i, int n) nogil:
    cdef __m128d summ = _mm_setzero_pd()
    cdef int tmp
    cdef __m128d a, b
    cdef double[2] double_arr
    cdef int j
    for j in range(0, n, 2):
        b = _mm_set_pd(v[j], v[j+1])
        a = _mm_set_pd(A(j, i), A(j + 1, i))
        summ = _mm_add_pd(summ, _mm_div_pd(b, a))
    _mm_store_pd(double_arr, summ)
    out[i] = double_arr[0] + double_arr[1]


cdef void mult_Av(vector[double] & v, vector[double] & out, int n) nogil:
    cdef int i, j
    for i in prange(n, nogil=True, num_threads=N_CPU):
        mult_Av_helper(v, out, i, n)

cdef void mult_Atv(vector[double] & v, vector[double] & out, int n) nogil:
    cdef int i, j
    cdef double summ
    cdef int tmp
    for i in prange(n, nogil=True, num_threads=N_CPU):
        mult_Atv_helper(v, out, i, n)

cdef void mult_AtAv(vector[double] & tmp, vector[double] & v, vector[double] & out, int n) nogil:
    mult_Av(v, tmp, n)
    mult_Atv(tmp, out, n)


cpdef main():
    cdef int n
    import sys
    n = int(sys.argv[1])
    if n % 2 == 1:
        n += 1
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


