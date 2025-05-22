#include <fstream>
#include <iostream>
#include <math.h>
template<double cval, double x, unsigned itercount, unsigned maxiter>
struct sqrt_impl{
    constexpr static double value = sqrt_impl<cval - (cval * cval - x) / (2.0 * cval), x, itercount + 1, maxiter>::value;
};
template<double cval, double x, unsigned itercount_equals_maxiter>
struct sqrt_impl<cval, x, itercount_equals_maxiter, itercount_equals_maxiter>{
    constexpr static double value = cval - (cval * cval - x) / (2.0 * cval);
};
template<typename T, T x>
struct sqrt_type{
    constexpr static double value = sqrt_impl<x / 2, x, 0, 20>::value;
};
double erfinv_taylor_horner(double x) {
    // Handle invalid input values
    if (x <= -1.0 || x >= 1.0) {
        return NAN; // Return Not a Number
    }

    // Constants (same as before)
    const double sqrt_pi = sqrt(M_PI);
    const double pi_3_2 = pow(M_PI, 1.5);
    const double pi_5_2 = pow(M_PI, 2.5);
    const double pi_7_2 = pow(M_PI, 3.5);
    const double pi_9_2 = pow(M_PI, 4.5);
    const double pi_11_2 = pow(M_PI, 5.5);

    const double c1 = sqrt_pi / 2;
    const double c2 = pi_3_2 / 24;
    const double c3 = 7 * pi_5_2 / 960;
    const double c4 = 127 * pi_7_2 / 80640;
    const double c5 = 4369 * pi_9_2 / 11612160;
    const double c6 = 34807 * pi_11_2 / 364953600;
    printf("c1 = %.3e\n", c1);
    printf("c2 = %.3e\n", c2);
    printf("c3 = %.3e\n", c3);
    printf("c4 = %.3e\n", c4);
    printf("c5 = %.3e\n", c5);
    printf("c6 = %.3e\n", c6);
    // Horner's Evaluation
    double result = c6;
    result = result * x * x + c5;
    result = result * x * x + c4;
    result = result * x * x + c3;
    result = result * x * x + c2;
    result = result * x * x + c1;
    result = result * x;
    return result;
}
double erfinv_winitzki(double x) {
    // Handle invalid input values
    if (x <= -1.0 || x >= 1.0) {
        return 7; // Return Not a Number
    }

    const double a1 = 0.27886807;
    const double a2 = -0.11352458;
    const double a3 = 0.04725462;
    const double a4 = -0.01784831;
    const double p = 0.88721;

    // Calculate intermediate values
    double z = sqrt(log(1 - x * x) / 2);
    double t = 1 / (1 + p * abs(z));

    // Approximate using polynomial
    double erf_inv = z * (a1 + t * (a2 + t * (a3 + t * a4)));

    // Adjust for sign and return
    return x > 0 ? erf_inv : -erf_inv;
}
double erfinv(double y) {
    constexpr double CENTRAL_RANGE = 0.7;
    constexpr double PI = M_PI;
    double x, z, num, dem; /*working variables */
    /* coefficients in rational expansion */
    double a[4] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
    double b[4] = {-2.118377725, 1.442710462, -0.329097515, 0.012229801};
    double c[4] = {-1.970840454, -1.624906493, 3.429567803, 1.641345311};
    double d[2] = {3.543889200, 1.637067800};
    if (fabs(y) > 1.0) return (NAN); /* This needs IEEE constant*/
    if (fabs(y) == 1.0) return ((copysign(1.0, y)) * 1e308);
    if (fabs(y) <= CENTRAL_RANGE) {
        z = y * y;
        num = (((a[3] * z + a[2]) * z + a[1]) * z + a[0]);
        dem = ((((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z + 1.0);
        x = y * num / dem;
    } else if ((fabs(y) > CENTRAL_RANGE) && (fabs(y) < 1.0)) {
        z = sqrt(-log((1.0 - fabs(y)) / 2.0));
        num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
        dem = (d[1] * z + d[0]) * z + 1.0;
        x = (copysign(1.0, y)) * num / dem;
    }
    /* Two steps of Newton-Raphson correction */
    x = x - (erf(x) - y) / ((2.0 / sqrt(M_PI)) * exp(-x * x));
    x = x - (erf(x) - y) / ((2.0 / sqrt_type<double, PI>::value) * exp(-x * x));

    return x;
}
double erfinv_nocorrec(double y) {
    double x, z, num, dem; /*working variables */
    /* coefficients in rational expansion */
    double a[4] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
    double b[4] = {-2.118377725, 1.442710462, -0.329097515, 0.012229801};
    double c[4] = {-1.970840454, -1.624906493, 3.429567803, 1.641345311};
    double d[2] = {3.543889200, 1.637067800};
    if (fabs(y) > 1.0) return (NAN); /* This needs IEEE constant*/
    if (fabs(y) == 1.0) return ((copysign(1.0, y)) * 1e308);
    if (fabs(y) <= CENTRAL_RANGE) {
        z = y * y;
        num = (((a[3] * z + a[2]) * z + a[1]) * z + a[0]);
        dem = ((((b[3] * z + b[2]) * z + b[1]) * z + b[0]) * z + 1.0);
        x = y * num / dem;
    } else if ((fabs(y) > CENTRAL_RANGE) && (fabs(y) < 1.0)) {
        z = sqrt(-log((1.0 - fabs(y)) / 2.0));
        num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
        dem = (d[1] * z + d[0]) * z + 1.0;
        x = (copysign(1.0, y)) * num / dem;
    }
    /* Two steps of Newton-Raphson correction */
    x = x - (erf(x) - y) / ((2.0 / sqrt(M_PI)) * exp(-x * x));
    //x = x - (erf(x) - y) / ((2.0 / sqrt(M_PI)) * exp(-x * x));

    return x;
}
int main() {
    std::cout.precision(30);
    //std::cout << sqrt_type<double, M_PI>::value << "\n";
    std::cout << M_PI << "\n";
    std::cout << sqrt_type<double, M_PI>::value * sqrt_type<double, M_PI>::value << "\n";
    //std::cout << std::sqrt(M_PI) << "\n";
    std::cout << std::sqrt(M_PI) * std::sqrt(M_PI) << "\n";
    std::ofstream ofile("erfinv.dat");
    for (double x = -0.999999; x < 0.999999; x += 1e-5) {
        double xv = tanh(100.0 * x);
        double y1 = erfinv_nocorrec(erf(xv));
        double y2 = abs(xv - y1);
        ofile << x << ' ' << y2 << '\n';
    }
}
