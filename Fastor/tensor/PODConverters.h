#ifndef PODCONVERTERS_H
#define PODCONVERTERS_H

FASTOR_INLINE T toscalar() const {
    //! Returns a scalar
    static_assert(size()==1,"ONLY TENSORS OF SIZE 1 CAN BE CONVERTED TO A SCALAR");
    return (*_data);
}

FASTOR_INLINE std::array<T,size()> toarray() const {
    //! Returns std::array
    std::array<T,size()> out;
    std::copy(_data,_data+size(),out.begin());
    return out;
}

FASTOR_INLINE std::vector<T> tovector() const {
    //! Returns std::vector
    std::vector<T> out(size());
    std::copy(_data,_data+size(),out.begin());
    return out;
}

#endif //PODCONVERTERS_H
