#ifndef INITIALIZER_LIST_CONSTRUCTORS_H
#define INITIALIZER_LIST_CONSTRUCTORS_H

// Initialiser list constructors
//----------------------------------------------------------------------------------------------------------//
    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false >
    FASTOR_INLINE Tensor(const std::initializer_list<U> &lst) {
        static_assert(sizeof...(Rest)==1,"TENSOR RANK MISMATCH WITH LIST-INITIALISER");
#if FASTOR_BOUNDS_CHECK
        FASTOR_ASSERT(pack_prod<Rest...>::value==lst.size(), "TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
#endif
        auto counter = 0;
        for (const auto &i: lst) {_data[counter] = i; counter++;}
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false >
    FASTOR_INLINE Tensor(const std::initializer_list<std::initializer_list<U>> &lst2d) {
        static_assert(sizeof...(Rest)==2,"TENSOR RANK MISMATCH WITH LIST-INITIALISER");
#ifndef NDEBUG
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        auto size_ = 0;
        FASTOR_ASSERT(M==lst2d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
        for (auto &lst: lst2d) {
            auto curr_size = lst.size();
            FASTOR_ASSERT(N==lst.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
            size_ += curr_size;
        }
        FASTOR_ASSERT(pack_prod<Rest...>::value==size_, "TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
#endif
        auto counter = 0;
        for (const auto &lst1d: lst2d) {
            for (const auto &i: lst1d) {
                _data[counter] = T(i);
                counter++;
            }
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false >
    FASTOR_INLINE Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<U>>> &lst3d) {
        static_assert(sizeof...(Rest)==3,"TENSOR RANK MISMATCH WITH LIST-INITIALISER");
#ifndef NDEBUG
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        constexpr FASTOR_INDEX P = get_value<3,Rest...>::value;
        auto size_ = 0;
        FASTOR_ASSERT(M==lst3d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
        for (const auto &lst2d: lst3d) {
            FASTOR_ASSERT(N==lst2d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
            for (const auto &lst: lst2d) {
                const auto curr_size = lst.size();
                FASTOR_ASSERT(P==lst.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
                size_ += curr_size;
            }
        }
        FASTOR_ASSERT(pack_prod<Rest...>::value==size_, "TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
#endif
        auto counter = 0;
        for (const auto &lst2d: lst3d) {
            for (const auto &lst1d: lst2d) {
                for (const auto &i: lst1d) {
                    _data[counter] = i;
                    counter++;
                }
            }
        }
    }

    template<typename U=T, enable_if_t_<is_primitive_v_<U>,bool> = false >
    FASTOR_INLINE Tensor(const std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<U>>>> &lst4d) {
        static_assert(sizeof...(Rest)==4,"TENSOR RANK MISMATCH WITH LIST-INITIALISER");
#ifndef NDEBUG
        constexpr FASTOR_INDEX M = get_value<1,Rest...>::value;
        constexpr FASTOR_INDEX N = get_value<2,Rest...>::value;
        constexpr FASTOR_INDEX P = get_value<3,Rest...>::value;
        constexpr FASTOR_INDEX Q = get_value<4,Rest...>::value;
        auto size_ = 0;
        FASTOR_ASSERT(M==lst4d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
        for (const auto &lst3d: lst4d) {
            FASTOR_ASSERT(N==lst3d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
            for (const auto &lst2d: lst3d) {
                FASTOR_ASSERT(P==lst2d.size(),"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
                for (const auto &lst: lst2d) {
                    const auto curr_size = lst.size();
                    FASTOR_ASSERT(Q==curr_size,"TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
                    size_ += curr_size;
                }
            }
        }
        FASTOR_ASSERT(pack_prod<Rest...>::value==size_, "TENSOR SIZE MISMATCH WITH LIST-INITIALISER");
#endif
        auto counter = 0;
        for (const auto &lst3d: lst4d) {
            for (const auto &lst2d: lst3d) {
                for (const auto &lst1d: lst2d) {
                    for (const auto &i: lst1d) {
                        _data[counter] = i;
                        counter++;
                    }
                }
            }
        }
    }
//----------------------------------------------------------------------------------------------------------//

#endif // INITIALIZER_LIST_CONSTRUCTORS_H
