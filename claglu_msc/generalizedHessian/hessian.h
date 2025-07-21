#ifndef hessian_h
#define hessian_h

#include <functional>
#include "field.h"

using namespace Mesh;

enum DiffType {Centered, Forward, Backward, CenteredDeriv2};

//////////////////////////////////////////////////////////////
// Stencil definitions along a template specified dimension //
//////////////////////////////////////////////////////////////

// More stencils can be found at:
// `https://en.wikipedia.org/wiki/Finite_difference_coefficient`

template<Dim D, typename T, class Callable>
inline T centered_stencil(const Index &idx, const T &hInv, const Callable &F){
    return 0.5 * hInv * (- F(idx.get_shifted<D>(-1)) + F(idx.get_shifted<D>(1)));
}

// Compact version of the `centered_stencil` for the 2nd derivative along the same dimension
template<Dim D, typename T, class Callable>
inline T centered_stencil_deriv2(const Index &idx, const T &hInv, const Callable &F){
    return hInv * hInv * (F(idx.get_shifted<D>(-1)) - 2.0*F(idx) + F(idx.get_shifted<D>(1)));
}

template<Dim D, typename T, class Callable>
inline T forward_stencil(const Index &idx, const T &hInv, const Callable &F){
    return 0.5 * hInv * (-3.0*F(idx) + 4.0*F(idx.get_shifted<D>(1)) - F(idx.get_shifted<D>(2)));
}

template<Dim D, typename T, class Callable>
inline T backward_stencil(const Index &idx, const T &hInv, const Callable &F){
    return 0.5 * hInv * (3.0*F(idx) - 4.0*F(idx.get_shifted<D>(-1)) + F(idx.get_shifted<D>(-2)));
}


///////////////////////////////////////////////
// Specialization to chain stencil operators //
///////////////////////////////////////////////

// This only works if the container the operators are applied to has an overloaded `operator()` //
template<Dim D, typename T, DiffType Diff, class C>
class DiffOpChain {
    public: 
        DiffOpChain(Field<T>& field) : f_m(field), hInvVector_m(field.getInvMeshSpacing()), leftOp(field) {}
        
        const inline T operator()(Index idx) const {
            if constexpr      (Diff == DiffType::Centered) { return centered_stencil<D,T,C>(idx, hInvVector_m[D], leftOp); }
            else if constexpr (Diff == DiffType::Forward)  { return forward_stencil<D,T,C>(idx, hInvVector_m[D], leftOp); }
            else if constexpr (Diff == DiffType::Backward) { return backward_stencil<D,T,C>(idx, hInvVector_m[D], leftOp); }
            else if constexpr (Diff == DiffType::CenteredDeriv2) { return centered_stencil_deriv2<D,T,C>(idx, hInvVector_m[D], leftOp); }
        }

    private:
        Field<T>& f_m;
        vector3d_t hInvVector_m;
        C leftOp;
};

template<typename T, DiffType DiffX, DiffType DiffY, DiffType DiffZ>
class GeneralizedHessOp {
    public:
        GeneralizedHessOp(Field<T> &field) : f_m(field){
            // Define operators of Hessian

            // Define typedefs for innermost operators applied to Field<T> as they are identical on each row
            typedef DiffOpChain<Dim::X,T,DiffX,Field<T>> colOpX_t;
            typedef DiffOpChain<Dim::Y,T,DiffY,Field<T>> colOpY_t;
            typedef DiffOpChain<Dim::Z,T,DiffZ,Field<T>> colOpZ_t;
            
            // Row 1
            DiffOpChain<Dim::X,T,DiffX,colOpX_t> hess_xx(f_m);
            DiffOpChain<Dim::X,T,DiffX,colOpY_t> hess_xy(f_m);
            DiffOpChain<Dim::X,T,DiffX,colOpZ_t> hess_xz(f_m);

            gen_row1_m = [hess_xx, hess_xy, hess_xz, this](Index idx){
                return (xvector_m*hess_xx(idx) +
                        yvector_m*hess_xy(idx) +
                        zvector_m*hess_xz(idx));
            };

            // Row 2
            DiffOpChain<Dim::Y,T,DiffY,colOpX_t> hess_yx(f_m);
            DiffOpChain<Dim::Y,T,DiffY,colOpY_t> hess_yy(f_m);
            DiffOpChain<Dim::Y,T,DiffY,colOpZ_t> hess_yz(f_m);

            gen_row2_m = [hess_yx, hess_yy, hess_yz, this](Index idx){
                return (xvector_m*hess_yx(idx) +
                        yvector_m*hess_yy(idx) +
                        zvector_m*hess_yz(idx));
            };
            
            // Row 3
            DiffOpChain<Dim::Z,T,DiffZ,colOpX_t> hess_zx(f_m);
            DiffOpChain<Dim::Z,T,DiffZ,colOpY_t> hess_zy(f_m);
            DiffOpChain<Dim::Z,T,DiffZ,colOpZ_t> hess_zz(f_m);

            gen_row3_m = [hess_zx, hess_zy, hess_zz, this](Index idx){
                return (xvector_m*hess_zx(idx) +
                        yvector_m*hess_zy(idx) +
                        zvector_m*hess_zz(idx));
            };
        }
        
        // Compute Hessian of specific Index `idx`
        const inline matrix3d_t operator()(Index idx) const {
            vector3d_t row_1, row_2, row_3;
            matrix3d_t hess_matrix;
            hess_matrix.row(0) = gen_row1_m(idx);
            hess_matrix.row(1) = gen_row2_m(idx);
            hess_matrix.row(2) = gen_row3_m(idx);
            return hess_matrix;
        }

    private:
        Field<T>& f_m;
        std::function<vector3d_t(Index)> gen_row1_m, gen_row2_m, gen_row3_m;
        vector3d_t xvector_m = {1.0, 0.0, 0.0};
        vector3d_t yvector_m = {0.0, 1.0, 0.0};
        vector3d_t zvector_m = {0.0, 0.0, 1.0};
};

template<typename T, DiffType DiffX, DiffType DiffY, DiffType DiffZ>
class CenteredIPPLHessOp {
    public:
        CenteredIPPLHessOp(Field<T> &field) : f_m(field){}

        const inline matrix3d_t operator()(Index idx){
			size_t i = std::get<0>(idx.idx);
			size_t j = std::get<1>(idx.idx);
			size_t k = std::get<2>(idx.idx);

			vector3d_t hvector = f_m.getInvMeshSpacing();
			vector3d_t row_1, row_2, row_3;

            // branch 136
            row_1 << ((f_m(i+1,j,k) - 2.0*f_m(i,j,k) + f_m(i-1,j,k))*(hvector[0]*hvector[0])),
                     ((f_m(i+1,j+1,k) - f_m(i-1,j+1,k) - f_m(i+1,j-1,k) + f_m(i-1,j-1,k))*(0.25*hvector[0]*hvector[1])),
                     ((f_m(i+1,j,k+1) - f_m(i-1,j,k+1) - f_m(i+1,j,k-1) + f_m(i-1,j,k-1))*(0.25*hvector[0]*hvector[2]));

            row_2 << ((f_m(i+1,j+1,k) - f_m(i+1,j-1,k) - f_m(i-1,j+1,k) + f_m(i-1,j-1,k))*(0.25*hvector[1]*hvector[0])),
                    ((f_m(i,j+1,k) - 2.0*f_m(i,j,k) + f_m(i,j-1,k))*(hvector[1]*hvector[1])),
                    ((f_m(i,j+1,k+1) - f_m(i,j-1,k+1) - f_m(i,j+1,k-1) + f_m(i,j-1,k-1))*(0.25*hvector[1]*hvector[2]));

            row_3 << ((f_m(i+1,j,k+1) - f_m(i+1,j,k-1) - f_m(i-1,j,k+1) + f_m(i-1,j,k-1))*(0.25*hvector[2]*hvector[0])),
                    ((f_m(i,j+1,k+1) - f_m(i,j+1,k-1) - f_m(i,j-1,k+1) + f_m(i,j-1,k-1))*(0.25*hvector[2]*hvector[1])),
                    ((f_m(i,j,k+1) - 2.0*f_m(i,j,k) + f_m(i,j,k-1))*(hvector[2]*hvector[2]));

			matrix3d_t hess_matrix;
			hess_matrix.row(0) = row_1;
			hess_matrix.row(1) = row_2;
			hess_matrix.row(2) = row_3;
			return hess_matrix; 
        }

    private:
        Field<T>& f_m;
};

#endif // hessian_h
