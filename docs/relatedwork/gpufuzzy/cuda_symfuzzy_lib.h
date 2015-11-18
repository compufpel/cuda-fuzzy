#ifndef CUDA_SYMFUZZY_LIB_H
#define CUDA_SYMFUZZY_LIB_H

#include "cuda_interval_rounded_arith.h"

template<class T, int N>
class symfuzzy_gpu
{
public:
    __device__ __host__ symfuzzy_gpu();
    __device__ __host__ symfuzzy_gpu(T const & v);
    __device__ __host__ symfuzzy_gpu(T const & m, T const * r);
    
    __host__ void random(T const & a, T const & b, int const & s);
    
    __device__ __host__ symfuzzy_gpu<T, N> & operator=(fuzzy_gpu<T, N> const & x);
    
    __device__ __host__ T const & get_mp() const;
    __device__ __host__ T const & get_rad(int const & a) const;
    __device__ __host__ void set_mp(T const & midpoint);
    __device__ __host__ void set_rad(int const & i, T const & radius); 

private:
    T mp;
    T rad[N];
};

// Constructors
template<class T, int N> inline __device__ __host__
symfuzzy_gpu<T, N>::symfuzzy_gpu()
{
}

template<class T, int N> inline __device__ __host__
symfuzzy_gpu<T, N>::symfuzzy_gpu(T const & m, T const * r) :
    mp(m)
{
    for (int i = 0; i < N; i++)
        rad[i] = r[i];
}

template<class T, int N> inline __device__ __host__
symfuzzy_gpu<T, N>::symfuzzy_gpu(T const & v) :
    mp(v)
{
    for (int i = 0; i < N; i++)
        rad[i] = 0;
}

// Member functions
template<class T, int N> inline __host__
void symfuzzy_gpu<T, N>::random(T const & a, T const & b, int const & s)
{
    srand(s);
    mp = a + (b - a) * rand() / RAND_MAX;
    rad[0] = min(b - mp, mp - a) * rand() / RAND_MAX;
    for (int i = 1; i < N; i++)
        rad[i] = rad[i - 1] * rand() / RAND_MAX;
}

template<class T, int N> inline __device__ __host__
T const & symfuzzy_gpu<T, N>::get_mp() const
{
    return mp;
}

template<class T, int N> inline __device__ __host__
T const & symfuzzy_gpu<T, N>::get_rad(int const & a) const
{
    return rad[a];
}

template<class T, int N> inline __device__ __host__
void symfuzzy_gpu<T, N>::set_mp(T const & midpoint)
{
    mp = midpoint;
}

template<class T, int N> inline __device__ __host__
void symfuzzy_gpu<T, N>::set_rad(int const & i, T const & radius)
{
    rad[i] = radius;
}

// Stream operator
template<class T, int N> inline __host__
std::ostream & operator<<(std::ostream & out, symfuzzy_gpu<T, N> const & y)
{
    out << y.get_mp() << " +- [";
    for(int i = 0; i < N - 1; i++)
        out << y.get_rad(i) << " , ";
        out << y.get_rad(N - 1) << "]";
    return out;
}

// Arithmetic operations

// Unary operators
template<class T, int N> inline __device__
symfuzzy_gpu<T, N> const & operator+(symfuzzy_gpu<T, N> const & x)
{
    return x;
}

template<class T, int N> inline __device__
symfuzzy_gpu<T, N> operator-(symfuzzy_gpu<T, N> const & x)
{
    symfuzzy_gpu<T, N> temp;
    temp.set_mp(-x.get_mp());
    return temp;
}

// Binary operators
template<class T, int N> inline __device__
symfuzzy_gpu<T, N> operator+(symfuzzy_gpu<T, N> const & x, symfuzzy_gpu<T, N> const & y)
{
    symfuzzy_gpu<T, N> temp;
    rounded_arith<T> rnd;
    temp.set_mp(x.get_mp() + y.get_mp());	
    T rump = rnd.mul_up(rnd.eps(), fabs(temp.get_mp()));
    for (int i = 0; i < N; i++){
        temp.set_rad(i, rnd.add_up(x.get_rad(i), y.get_rad(i)));
        temp.set_rad(i, rnd.add_up(rump, temp.get_rad(i)));
    }
    return temp;
}

template<class T, int N> inline __device__
symfuzzy_gpu<T, N> operator-(symfuzzy_gpu<T, N> const & x, symfuzzy_gpu<T, N> const & y)
{
    symfuzzy_gpu<T, N> temp;
    rounded_arith<T> rnd;
    temp.set_mp(x.get_mp() - y.get_mp());	
    T rump = rnd.mul_up(rnd.eps(), fabs(temp.get_mp()));
    for (int i = 0; i < N; i++){
        temp.set_rad(i, rnd.add_up(x.get_rad(i), y.get_rad(i)));
        temp.set_rad(i, rnd.add_up(rump, temp.get_rad(i)));
    }
    return temp;
}

template<class T, int N> inline __device__
symfuzzy_gpu<T, N> operator*(symfuzzy_gpu<T, N> const & x, symfuzzy_gpu<T, N> const & y)
{    
    symfuzzy_gpu<T, N> temp;
    rounded_arith<T> rnd;
    temp.set_mp(x.get_mp() * y.get_mp());	
    T rump = rnd.mul_up(rnd.eps(), fabs(temp.get_mp()));
    T a = fabs(x.get_mp());
    T b = fabs(y.get_mp());
    for (int i = 0; i < N; i++){
        //temp.set_rad(i, rnd.add_up(rnd.add_up(rump, rnd.mul_up(rnd.add_up(fabs(x.get_mp()), x.get_rad(i)), y.get_rad(i))), rnd.mul_up(fabs(y.get_mp()), x.get_rad(i))));
        temp.set_rad(i, rnd.add_up(a, x.get_rad(i)));
        temp.set_rad(i, rnd.mul_up(temp.get_rad(i), y.get_rad(i)));
        b = rnd.mul_up(b, x.get_rad(i));
        temp.set_rad(i, rnd.add_up(temp.get_rad(i), b));
        temp.set_rad(i, rnd.add_up(temp.get_rad(i), rump));
    }
    return temp;
}
#endif
