#include <fstream>
#include "hessian.h"

template<class HessOp>
matrix3d_t hessError(Field<matrix3d_t> &field_exact, HessOp &hess, bool is_rel_error){
    Mesh::vector3i_t N_ext =  field_exact.getN_ext();
    size_t nghost = field_exact.getNghost();

    matrix3d_t appr_error_matrix = matrix3d_t::Zero();
    matrix3d_t orig_sum_matrix = matrix3d_t::Zero();

    for(size_t i = 5*nghost; i < N_ext[0]-5*nghost; ++i){
        for(size_t j = 5*nghost; j < N_ext[1]-5*nghost; ++j){
            for(size_t k = 5*nghost; k < N_ext[2]-5*nghost; ++k){

                matrix3d_t approx_hess = hess(Index({i,j,k}, N_ext));

                for(size_t dim0 = 0; dim0 < 3; ++dim0){
                    for(size_t dim1 = 0; dim1 < 3; ++dim1){
                        appr_error_matrix(dim0, dim1) += std::pow(field_exact(i,j,k)(dim0,dim1) - approx_hess(dim0, dim1), 2);
                        orig_sum_matrix(dim0, dim1) += std::pow(field_exact(i,j,k)(dim0, dim1), 2);
                    }
                }
            }
        }
    }

    orig_sum_matrix = orig_sum_matrix.cwiseSqrt();
    
    // Avoid division by zero when ground truth value is `0.0`
    if (is_rel_error) {
        appr_error_matrix = appr_error_matrix.cwiseSqrt();
        appr_error_matrix = (appr_error_matrix.array() < 1e-15 || orig_sum_matrix.array() < 1e-15).select(matrix3d_t::Zero(),appr_error_matrix.cwiseQuotient(orig_sum_matrix));
    }

    return appr_error_matrix;
}

auto format_matrix3d(matrix3d_t &matrix, std::ofstream &file){
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", ",");
    return matrix.format(CommaInitFmt);
}
