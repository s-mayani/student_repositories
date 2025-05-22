#include "hessian.h"
#include "convergence.h"

int main(){
    using namespace Mesh;
    /*
    size_t nghost = 1;
    vector3d_t h_vec{1e-2, 1e-2, 1e-2};

	// Scalar Fields
    Field<double> field(h_vec);
    Field<double> field_result(h_vec);
    Field<double> field_exact(h_vec);

    vector3i_t N = field.getN();
    vector3i_t N_ext = field.getN_ext();

    // Initialize
    field.init_x2y2z2();
    field_exact.init_x2y2z2_deriv();


	////////////////////////////////////////////
	// Test individual Differential Operators //
	////////////////////////////////////////////

    // Construct differential operator
    DiffOpChain<Dim::X, double, DiffType::Centered, 
                DiffOpChain<Dim::Z, double, DiffType::Centered,
                            Field<double>>> xzDiff(field);

    double error = 0.0;
    for(size_t i = 2*nghost; i < N_ext[0]-2*nghost; ++i){
        for(size_t j = 2*nghost; j < N_ext[1]-2*nghost; ++j){
            for(size_t k = 2*nghost; k < N_ext[2]-2*nghost; ++k){
                // Apply stencil
                double approx_value = xzDiff(Index({i,j,k},N_ext));

                field_result(i,j,k) = approx_value;

                // Compute L2 error
                error += pow(field_exact(i,j,k) - approx_value, 2);
            }
        }
    }

    std::cout << "Absolute L2 Error of chained differential operators: " << std::setw(10) << sqrt(error) << std::endl;


	/////////////////////////////////////
	// Test Hessian Matrix computation //
	/////////////////////////////////////
    
	// Matrix Fields
	Field<matrix3d_t> mfield(h_vec);
	Field<matrix3d_t> mfield_result(h_vec);
	Field<matrix3d_t> mfield_exact(h_vec);

    N = field.getN();
    N_ext = field.getN_ext();
    
    // Initialize
    field.init_xyz();
    bool gaussian_ic = 0;
	mfield_exact.initHess(gaussian_ic);
    
	// Construct Hessian operator
    GeneralizedHessOp<double,DiffType::Centered,DiffType::Centered,DiffType::Centered> hessOp(field);
    //GeneralizedHessOp<double,DiffType::Forward,DiffType::Forward,DiffType::Forward> hessOp(field);
    //GeneralizedHessOp<double,DiffType::Backward,DiffType::Backward,DiffType::Backward> hessOp(field);
 
    matrix3d_t hess_error;
    hess_error.setZero();
    double avg_hess_error = 0.0;
    for(size_t i = 5*nghost; i < N_ext[0]-5*nghost; ++i){
        for(size_t j = 5*nghost; j < N_ext[1]-5*nghost; ++j){
            for(size_t k = 5*nghost; k < N_ext[2]-5*nghost; ++k){

                // Apply Operator
                matrix3d_t approx_hess = hessOp(Index({i,j,k}, N_ext));
                mfield_result(i,j,k) = approx_hess;

                for(size_t dim0 = 0; dim0 < 3; ++dim0){
                    for(size_t dim1 = 0; dim1 < 3; ++dim1){
                        hess_error(dim0, dim1) += pow(mfield_exact(i,j,k)(dim0,dim1) - approx_hess(dim0, dim1), 2);
                    }
                }
            }
        }
    }


    std::cout << "\n=================================================" << std::endl;
    std::cout << "Error Hessian Operator:\n" << hess_error.cwiseSqrt() << std::endl;
    std::cout << "\nAverage Error Hessian Operator: " << hess_error.sum()/hess_error.size() << std::endl;
    */
	
	/////////////////////////////////////
	// Test Hessian Matrix Convergence //
	/////////////////////////////////////

    std::cout << "\n=================================================" << std::endl;
    std::cout << "Convergence Test Hessian Operator:\n\n" << std::endl;

	double h_begin = 5e-2;

    // How many mesh-widths to test
    size_t N_test = 8;

    std::vector<matrix3d_t> rel_error(N_test);
    std::vector<double> mesh_widths(N_test);
    std::vector<int> mesh_sizes(N_test);
    double h = h_begin;
    vector3d_t h_vec;

    for(int i = 0; i < N_test; ++i){
        mesh_widths[i] = h;
        h_vec << h, h, h;

        // Matrix Fields
        Field<double> field_convtest(h_vec);
        Field<matrix3d_t> mfield_exact_convtest(h_vec);

        // Initialize
        bool is_gaussian_ic = true;
        if (is_gaussian_ic) {
            field_convtest.init_gaussian();
        }
        else {
            field_convtest.init_xyz();
        }

        mfield_exact_convtest.initHess(is_gaussian_ic);

        // Hessian Operator
        //GeneralizedHessOp<double,DiffType::Centered,DiffType::Forward,DiffType::Backward> hessOp_convtest(field_convtest);
        CenteredIPPLHessOp<double,DiffType::Centered,DiffType::Forward,DiffType::Backward> hessOp_convtest(field_convtest);

        vector3i_t N = field_convtest.getN();
        vector3i_t N_ext = field_convtest.getN_ext();
        mesh_sizes[i] = N[0];

        rel_error[i] = hessError(mfield_exact_convtest, hessOp_convtest, is_gaussian_ic);

        std::cout << "[" << N[0] << "]^3 ; h = " << h << ": " << rel_error[i].sum()/9.0 << std::endl;

        h /= 1.4;
    }

    // Filestream to write matrix to
    char fname[] = "rel_error.csv";
    std::ofstream file(fname);
    Eigen::IOFormat CommaInitFmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ",", ",");
    // Stream Header
    file << "N,avg. H_f";
    for(int i = 0; i < 3; ++i) { for (int j = 0; j < 3; ++j) {
        file << ",(" << i << ";" << j << ")";
    }}
    file << '\n';

    for (int i = 0; i < N_test; ++i){
        file << mesh_sizes[i] << "," << rel_error[i].sum()/9.0 << "," << rel_error[i].format(CommaInitFmt) <<  "\n";
    }

    file.close();

    return 0;
}
