#ifndef CUDA_FUZZY_LIB_H
#define CUDA_FUZZY_LIB_H

#include "cuda_interval_lib.h"

template<class T, int N>
class symfuzzy_gpu;

template<class T, int N>
class fuzzy_gpu
{
public:
    __device__ __host__ fuzzy_gpu();
    __device__ __host__ fuzzy_gpu(T const & v);
    __device__ __host__ fuzzy_gpu(T const * l, T const * u);
    
    __device__ __host__ fuzzy_gpu<T, N> & operator=(symfuzzy_gpu<T, N> const & x);
    
    __device__ __host__ interval_gpu<T> const & get_cut(int const & a) const;
    __device__ __host__ void set_cut(int const & a, interval_gpu<T> const & i);    

private:
    interval_gpu<T> cut[N];
};

// Constructors
template<class T, int N> inline __device__ __host__
fuzzy_gpu<T, N>::fuzzy_gpu()
{
}

template<class T, int N> inline __device__ __host__
fuzzy_gpu<T, N>::fuzzy_gpu(T const * l, T const * u)
{
    for (int i = 0; i < N; i++)
	cut[i] = interval_gpu<T>(l[i], u[i]);
}

template<class T, int N> inline __device__ __host__
fuzzy_gpu<T, N>::fuzzy_gpu(T const & v)
{
    for (int i = 0; i < N; i++)
	cut[i] = interval_gpu<T>(v);
}

// Member functions

// Assignment operator
template<class T, int N> inline __device__ __host__
fuzzy_gpu<T, N> & fuzzy_gpu<T, N>::operator=(symfuzzy_gpu<T, N> const & x)
{
    for (int i = 0; i < N; i++)
    	this->set_cut(i, interval_gpu<T>(x.get_mp() - x.get_rad(i), x.get_mp() + x.get_rad(i)));
    return *this;
}

template<class T, int N> inline __device__ __host__
interval_gpu<T> const & fuzzy_gpu<T, N>::get_cut(int const & a) const
{
    return cut[a];
}

template<class T, int N> inline __device__ __host__
void fuzzy_gpu<T, N>::set_cut(int const & a, interval_gpu<T> const & i)
{
    cut[a] = i;
}

// Stream operator
template<class T, int N> inline __host__
std::ostream & operator<<(std::ostream & out, fuzzy_gpu<T, N> const & y)
{
    for(int i = 0; i < N - 1; i++)
	out << "[" << y.get_cut(i).lower() << " , " << y.get_cut(i).upper() << "] -- ";
	out << "[" << y.get_cut(N - 1).lower() << " , " << y.get_cut(N - 1).upper() << "]";
    return out;
}

// Arithmetic operations

// Unary operators
template<class T, int N> inline __device__
fuzzy_gpu<T, N> const & operator+(fuzzy_gpu<T, N> const & x)
{
    return x;
}

template<class T, int N> inline __device__
fuzzy_gpu<T, N> operator-(fuzzy_gpu<T, N> const & x)
{
    fuzzy_gpu<T, N> temp;
    for (int i = 0; i < N; i++)
	temp.set_cut(i, -x.get_cut(i));
    return temp;
}

// Binary operators
template<class T, int N> inline __device__
fuzzy_gpu<T, N> operator+(fuzzy_gpu<T, N> const & x, fuzzy_gpu<T, N> const & y)
{
    fuzzy_gpu<T, N> temp;
    for (int i = 0; i < N; i++)
        temp.set_cut(i, x.get_cut(i) + y.get_cut(i));
    return temp;
}

template<class T, int N> inline __device__
fuzzy_gpu<T, N> operator-(fuzzy_gpu<T, N> const & x, fuzzy_gpu<T, N> const & y)
{
    fuzzy_gpu<T, N> temp;
    for (int i = 0; i < N; i++)
        temp.set_cut(i, x.get_cut(i) - y.get_cut(i));
    return temp;
}

template<class T, int N> inline __device__
fuzzy_gpu<T, N> operator*(fuzzy_gpu<T, N> const & x, fuzzy_gpu<T, N> const & y)
{
    fuzzy_gpu<T, N> temp;
    for (int i = 0; i < N; i++)
        temp.set_cut(i, x.get_cut(i) * y.get_cut(i));
    return temp;
}
#endif
