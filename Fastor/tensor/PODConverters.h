#ifndef PODCONVERTERS_H
#define PODCONVERTERS_H

FASTOR_INLINE T toscalar() const {
    //! Returns a scalar
    static_assert(Size==1,"ONLY TENSORS OF SIZE 1 CAN BE CONVERTED TO SCALAR");
    return _data[0];
}

FASTOR_INLINE std::array<T,Size> toarray() const {
    //! Returns std::array
    std::array<T,Size> out;
    std::copy(_data,_data+Size,out.begin());
    return out;
}

FASTOR_INLINE std::vector<T> tovector() const {
    //! Returns std::vector
    std::vector<T> out(Size);
    std::copy(_data,_data+Size,out.begin());
    return out;
}

#endif //PODCONVERTERS_H