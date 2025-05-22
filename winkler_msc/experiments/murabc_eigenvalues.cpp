#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <sciplot/sciplot.hpp>
#include <immintrin.h>
using namespace Eigen;
//using index = std::int64_t;
template<typename T>
auto constexpr sq(T x){
    return x * x;
}
/**
 * @brief Convert an Eigen vector to a std::valarray.
 * 
 * @tparam T The type of elements in the Eigen vector.
 * @param eigenVector The Eigen vector to convert.
 * @return std::valarray<T> The converted std::valarray.
 */
template<typename T>
std::valarray<T> eigenVectorToValarray(const Eigen::Matrix<T, Eigen::Dynamic, 1>& eigenVector) {
    std::valarray<T> valarray(eigenVector.size());
    for (int i = 0; i < eigenVector.size(); ++i) {
        valarray[i] = eigenVector[i];
    }
    return valarray;
}
/**
 * @brief Convert a std::vector<T> to a std::valarray<T>.
 * 
 * @tparam T The type of elements in the vector and valarray.
 * @param vec The vector to convert.
 * @return std::valarray<T> The converted valarray.
 */
template<typename T>
std::valarray<T> vectorToValarray(const std::vector<T>& vec) {
    return std::valarray<T>(vec.data(), vec.size());
}
/**
 * @brief Compute the least square error Ax = 0 using Cholesky decomposition.
 * 
 * This function computes the least square error Ax = 0 using Cholesky decomposition.
 * 
 * @param A The matrix A.
 * @return The least square error solution vector x.
 */
//Eigen::VectorXd leastSquares(const Eigen::MatrixXd& A, const Eigen::VectorXd& rem) {
//    return A.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(rem);
//}
//VectorXd leastSquareError(const MatrixXd A, const VectorXd rem) {
//    // Compute A^T * A
//    MatrixXd ATA = A.transpose() * A;
//
//    // Perform Cholesky decomposition on ATA
//    LLT<MatrixXd> llt(ATA);
//    VectorXd rems = A.transpose() * rem;
//    VectorXd x = llt.solve(rems);
//
//    return x;
//}
//size_t realReconstructionWeights(const MatrixXcd& eigenvectors){
//    const size_t N = eigenvectors.cols();
//    for(size_t n = 2;n < N;n++){
//        MatrixXd chunk = eigenvectors.block(0, eigenvectors.cols() - n + 1, eigenvectors.rows(), n - 1).imag();
//        VectorXd lsq = leastSquares(chunk, eigenvectors.col(eigenvectors.cols() - n).imag());
//        std::cout << (chunk * lsq).norm() << "\n";
//    }
//    return N;
//}
template<typename scalar>
MatrixX<scalar> update_matrix_mur(double dx, double dt, int n){
    MatrixX<scalar> update_matrix(2 * n + 2, 2 * n + 2);
    const scalar a1 = scalar(2) * (scalar(1) - sq(dt / dx));
    const scalar a2 = sq(dt / dx);
    const scalar a8 = sq(dt);
    update_matrix.setZero();
    auto An_index = [n](int i){
        return i + 1;
    };
    auto Anm1_index = [n](int i){
        return i + n + 2;
    };

    for(int i = 0;i < n;i++){
        update_matrix(An_index(i), Anm1_index(i)) = -1;
        update_matrix(An_index(i), An_index(i))   = a1;
        update_matrix(An_index(i), An_index(i+1)) = a2;
        update_matrix(An_index(i), An_index(i-1)) = a2;
        update_matrix(Anm1_index(i), An_index(i)) = 1;
    }
    update_matrix(An_index(-1), An_index(-1))  = 1;
    update_matrix(An_index(-1), An_index(-1)) -= dt/dx;
    update_matrix(An_index(-1), An_index(0))  += dt/dx;
    update_matrix(An_index(n), An_index(n)) = 1;
    update_matrix(An_index(n), An_index(n)) -= dt/dx;
    update_matrix(An_index(n), An_index(n - 1)) += dt/dx;
    return update_matrix;
}

template<typename scalar>
MatrixX<scalar> update_matrix_fallahi(double dx, double dt, int n){
    MatrixX<scalar> update_matrix(2 * n + 2, 2 * n + 2);
    const scalar a1 = scalar(2) * (scalar(1) - sq(dt / dx));
    const scalar a2 = sq(dt / dx);
    const scalar a8 = sq(dt);
    const scalar beta0 = (dt - dx) / (dt + dx);
    const scalar beta1 = (2 * dx) / (dt + dx);
    constexpr scalar beta2 = -1;
    update_matrix.setZero();
    auto An_index = [n](int i){
        return i;
    };
    auto Anm1_index = [n](int i){
        return i + n;
    };
    for(int i = 1;i < n - 1;i++){
        update_matrix(An_index(i), Anm1_index(i)) = -1;
        update_matrix(An_index(i), An_index(i))   = a1;
        update_matrix(An_index(i), An_index(i+1)) = a2;
        update_matrix(An_index(i), An_index(i-1)) = a2;
        update_matrix(Anm1_index(i), An_index(i)) = 1;
    }
    
    update_matrix(An_index(0), Anm1_index(0)) = beta0;
    update_matrix(An_index(0), An_index(0)) = beta1;
    update_matrix(An_index(0), Anm1_index(1)) = beta2;
    update_matrix(An_index(0), An_index(1)) = beta1;
    update_matrix.row(An_index(0)) += update_matrix.row(An_index(1)) * beta0;
    update_matrix(Anm1_index(0), An_index(0)) = 1;

    update_matrix(An_index(n - 1), Anm1_index(n - 1)) = beta0;
    update_matrix(An_index(n - 1), An_index(n - 1))   = beta1;
    update_matrix(An_index(n - 1), Anm1_index(n - 2)) = beta2;
    update_matrix(An_index(n - 1), An_index(n - 2))   = beta1;
    update_matrix.row(An_index(n - 1)) += update_matrix.row(An_index(n - 2)) * beta0;
    update_matrix(Anm1_index(n - 1), An_index(n - 1)) = 1;
    return update_matrix;
}
int main(){
    std::ofstream evlog("evlog.txt");
    sciplot::Plot2D plot;
    plot.xlabel("Grid points");
    plot.ylabel("Steps");
    plot.xtics().logscale(2);
    plot.ytics().logscale(2);
    plot.ytics().increment(2);
    plot.size(1200, 800);
    {
        const int n = 500;
        MatrixXd update_matrix = update_matrix_fallahi<double>(1.0, 0.6, n);
        auto An_index = [n](int i){
            return i;
        };
        auto Anm1_index = [n](int i){
            return i + n;
        };
        VectorXd state(2 * n);
        std::ofstream initial("initial.txt");
        for(int i = 0;i < n;i++){
            double x = double(i) / (n - 1);
            double garg = -0.5 * sq(x - 0.5) / sq(0.1);
            state(An_index  (i)) = std::exp(garg);
            state(Anm1_index(i)) = std::exp(garg);   
        }
        for(int i = 0; i < n / 2;i++){
            state = update_matrix * state;
        }
        VectorXd state2 = state;
        for(int i = 0; i < n / 2;i++){
            state2 = update_matrix * state2;
        }
        VectorXd state3 = state2;
        for(int i = 0; i < n / 2;i++){
            state3 = update_matrix * state3;
        }
        for(int i = 0;i < n;i++){
            double x = double(i) / (n - 1);
            double garg = -0.5 * sq(x - 0.5) / sq(0.1);
            initial << x << " " << std::exp(garg) << " " << state(An_index(i)) << " " << state2(An_index(i)) << " " << state3(An_index(i)) << "\n";
        }
        initial.close();
    }
    // Set the x and y ranges
    std::map<int, std::vector<double>> v;
    for(int n = 16;n <= 200;n *= 1.2){
        using scalar = double;
        constexpr double dx = 1.0;
        constexpr double dt = 0.5;
        auto An_index = [n](int i){
            return i;
        };
        auto Anm1_index = [n](int i){
            return i + n;
        };
        
        MatrixXd update_matrix = update_matrix_fallahi<scalar>(dx, dt, n);
        EigenSolver<MatrixXd> solver(update_matrix);
        std::cout << solver.eigenvalues().cwiseAbs().maxCoeff() << "\n";
        std::vector<std::pair<Index, dcomplex>> evs(solver.eigenvalues().rows());
        for(Index i = 0;i < evs.size();i++){
            evs[i] = {i, solver.eigenvalues()(i)};
        }
        std::sort(evs.begin(), evs.end(), [](std::pair<Index, dcomplex> x, std::pair<Index, dcomplex> y){
            return std::abs(x.second) > std::abs(y.second);
        });
        MatrixXcd Tm1 = solver.eigenvectors().inverse();
        VectorXd highestmode(Tm1.rows());
        highestmode.setZero();
        for(size_t i = 2;i < n - 2;i++){
            highestmode(An_index(i)) = double(i & 1);
            highestmode(Anm1_index(i)) = double((i) & 1);
        }
        highestmode.array() -= highestmode.mean();
        VectorXd highest_mode_updated = highestmode;
        size_t steps = 0;
        do{
            highest_mode_updated = VectorXd(update_matrix * highest_mode_updated);
            highest_mode_updated.array() -= highest_mode_updated.mean();
            steps++;
            //std::cout << "highest_mode_updated.norm(): " << highest_mode_updated.norm() / highestmode.norm() << "\n";
            if(steps > 1000000){
                std::cout << highest_mode_updated.transpose() << "\n";
                //std::terminate();
                break;
            }
        } while(highest_mode_updated.norm() / highestmode.norm() >= 0.5);
        double norm_multiplier_of_actual_highestmode = (update_matrix * highestmode).norm() / highestmode.norm();
        VectorXcd highestmode_in_eigenbasis = Tm1 * highestmode;
        //std::cout << solver.eigenvalues().cwiseAbs()(2 * n - 1) << "\n";
        double evabs = std::abs(evs[1].second);// = solver.eigenvalues().cwiseAbs()(2 * n - 1);
        //realReconstructionWeights(solver.eigenvectors());
        //std::cout << "Steps required " << steps << "\n";
        v[0].push_back(n);
        //v[1].push_back(std::log(0.5) / std::log(std::abs(evs[1].second)));
        v[2].push_back(std::log(0.5) / std::log(std::abs(evs[9].second)));
        //v[3].push_back(std::log(0.5) / std::log(std::abs(evs[49].second)));
        v[4].push_back(steps);

        //std::cout << solver.eigenvectors().col(evs[0].first) << "\n";
        //std::cout << solver.eigenvectors().col(evs[1].first) << "\n";

    }
    sciplot::Plot2D plot2;
    std::ofstream ofile("evs.txt");
    plot2.xlabel("Grid points");
    plot2.ylabel("Steps");
    plot2.xtics().logscale(10).format("10^{%L}");;
    plot2.ytics().logscale(10).format("10^{%L}");;
    plot2.ytics().increment(10);
    plot2.size(1200, 800);
    //std::cout << "prepr: \n" << plot.repr() << "\n";
    plot.drawCurve(v[0], v[1]).label("Second largest Eigenvalue");
    plot.drawCurve(v[0], v[2]).label("Tenth largest Eigenvalue");
    //plot.drawCurve(v[0], v[3]).label("50th largest Eigenvalue");
    plot2.drawCurve(v[0], v[4]).label("Amount of steps actually required");
    for(int i = 0;i < v[0].size();i++){
        ofile << v[0][i] << ' ' << v[2][i] << ' ' << v[4][i] << "\n";
    }
    ofile.flush();
    plot.grid().show();
    plot.grid().lineWidth(2);
    plot2.grid().show();
    plot2.grid().lineWidth(2);
    plot.legend().atBottomRight();
    plot.fontName("CMU Serif");
    plot.fontSize(24);
    plot.ytics().fontSize(22);
    plot.xtics().fontSize(22);
    plot2.fontSize(18);
    plot2.ytics().fontSize(22);
    plot2.xtics().fontSize(22);
    sciplot::Figure fig = {{plot}, {plot2}};
    fig.title("Timesteps required to halve high amplitudes");
    //fig.fontSize(25);
    // Create canvas to hold figure
    sciplot::Canvas canvas = {{fig}};
    canvas.fontSize(30);

    canvas.size(1200, 1600);
    canvas.save("evs.svg");
    //for(int i = 0;i < 100;i++){
    //    state = VectorXd(update_matrix * state);
    //}
    //std::ofstream final("final.txt");
    //for(int i = 0;i < n;i++){
    //    scalar x = scalar(i) / (n - 1);
    //    final << x << ' ' << state(An_index(i)) << "\n";
    //}
    //std::cout << update_matrix << "\n";
}