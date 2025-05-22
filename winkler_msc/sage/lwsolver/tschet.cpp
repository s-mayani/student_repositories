#include <ProgramOptions.hxx>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <random>
#include <xoshiro.hpp>
#include "jet.hpp"
#include <chrono>
#include <fstream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
constexpr double turbo_cm[256][3] = {
  {0.18995,0.07176,0.23217},
  {0.19483,0.08339,0.26149},
  {0.19956,0.09498,0.29024},
  {0.20415,0.10652,0.31844},
  {0.20860,0.11802,0.34607},
  {0.21291,0.12947,0.37314},
  {0.21708,0.14087,0.39964},
  {0.22111,0.15223,0.42558},
  {0.22500,0.16354,0.45096},
  {0.22875,0.17481,0.47578},
  {0.23236,0.18603,0.50004},
  {0.23582,0.19720,0.52373},
  {0.23915,0.20833,0.54686},
  {0.24234,0.21941,0.56942},
  {0.24539,0.23044,0.59142},
  {0.24830,0.24143,0.61286},
  {0.25107,0.25237,0.63374},
  {0.25369,0.26327,0.65406},
  {0.25618,0.27412,0.67381},
  {0.25853,0.28492,0.69300},
  {0.26074,0.29568,0.71162},
  {0.26280,0.30639,0.72968},
  {0.26473,0.31706,0.74718},
  {0.26652,0.32768,0.76412},
  {0.26816,0.33825,0.78050},
  {0.26967,0.34878,0.79631},
  {0.27103,0.35926,0.81156},
  {0.27226,0.36970,0.82624},
  {0.27334,0.38008,0.84037},
  {0.27429,0.39043,0.85393},
  {0.27509,0.40072,0.86692},
  {0.27576,0.41097,0.87936},
  {0.27628,0.42118,0.89123},
  {0.27667,0.43134,0.90254},
  {0.27691,0.44145,0.91328},
  {0.27701,0.45152,0.92347},
  {0.27698,0.46153,0.93309},
  {0.27680,0.47151,0.94214},
  {0.27648,0.48144,0.95064},
  {0.27603,0.49132,0.95857},
  {0.27543,0.50115,0.96594},
  {0.27469,0.51094,0.97275},
  {0.27381,0.52069,0.97899},
  {0.27273,0.53040,0.98461},
  {0.27106,0.54015,0.98930},
  {0.26878,0.54995,0.99303},
  {0.26592,0.55979,0.99583},
  {0.26252,0.56967,0.99773},
  {0.25862,0.57958,0.99876},
  {0.25425,0.58950,0.99896},
  {0.24946,0.59943,0.99835},
  {0.24427,0.60937,0.99697},
  {0.23874,0.61931,0.99485},
  {0.23288,0.62923,0.99202},
  {0.22676,0.63913,0.98851},
  {0.22039,0.64901,0.98436},
  {0.21382,0.65886,0.97959},
  {0.20708,0.66866,0.97423},
  {0.20021,0.67842,0.96833},
  {0.19326,0.68812,0.96190},
  {0.18625,0.69775,0.95498},
  {0.17923,0.70732,0.94761},
  {0.17223,0.71680,0.93981},
  {0.16529,0.72620,0.93161},
  {0.15844,0.73551,0.92305},
  {0.15173,0.74472,0.91416},
  {0.14519,0.75381,0.90496},
  {0.13886,0.76279,0.89550},
  {0.13278,0.77165,0.88580},
  {0.12698,0.78037,0.87590},
  {0.12151,0.78896,0.86581},
  {0.11639,0.79740,0.85559},
  {0.11167,0.80569,0.84525},
  {0.10738,0.81381,0.83484},
  {0.10357,0.82177,0.82437},
  {0.10026,0.82955,0.81389},
  {0.09750,0.83714,0.80342},
  {0.09532,0.84455,0.79299},
  {0.09377,0.85175,0.78264},
  {0.09287,0.85875,0.77240},
  {0.09267,0.86554,0.76230},
  {0.09320,0.87211,0.75237},
  {0.09451,0.87844,0.74265},
  {0.09662,0.88454,0.73316},
  {0.09958,0.89040,0.72393},
  {0.10342,0.89600,0.71500},
  {0.10815,0.90142,0.70599},
  {0.11374,0.90673,0.69651},
  {0.12014,0.91193,0.68660},
  {0.12733,0.91701,0.67627},
  {0.13526,0.92197,0.66556},
  {0.14391,0.92680,0.65448},
  {0.15323,0.93151,0.64308},
  {0.16319,0.93609,0.63137},
  {0.17377,0.94053,0.61938},
  {0.18491,0.94484,0.60713},
  {0.19659,0.94901,0.59466},
  {0.20877,0.95304,0.58199},
  {0.22142,0.95692,0.56914},
  {0.23449,0.96065,0.55614},
  {0.24797,0.96423,0.54303},
  {0.26180,0.96765,0.52981},
  {0.27597,0.97092,0.51653},
  {0.29042,0.97403,0.50321},
  {0.30513,0.97697,0.48987},
  {0.32006,0.97974,0.47654},
  {0.33517,0.98234,0.46325},
  {0.35043,0.98477,0.45002},
  {0.36581,0.98702,0.43688},
  {0.38127,0.98909,0.42386},
  {0.39678,0.99098,0.41098},
  {0.41229,0.99268,0.39826},
  {0.42778,0.99419,0.38575},
  {0.44321,0.99551,0.37345},
  {0.45854,0.99663,0.36140},
  {0.47375,0.99755,0.34963},
  {0.48879,0.99828,0.33816},
  {0.50362,0.99879,0.32701},
  {0.51822,0.99910,0.31622},
  {0.53255,0.99919,0.30581},
  {0.54658,0.99907,0.29581},
  {0.56026,0.99873,0.28623},
  {0.57357,0.99817,0.27712},
  {0.58646,0.99739,0.26849},
  {0.59891,0.99638,0.26038},
  {0.61088,0.99514,0.25280},
  {0.62233,0.99366,0.24579},
  {0.63323,0.99195,0.23937},
  {0.64362,0.98999,0.23356},
  {0.65394,0.98775,0.22835},
  {0.66428,0.98524,0.22370},
  {0.67462,0.98246,0.21960},
  {0.68494,0.97941,0.21602},
  {0.69525,0.97610,0.21294},
  {0.70553,0.97255,0.21032},
  {0.71577,0.96875,0.20815},
  {0.72596,0.96470,0.20640},
  {0.73610,0.96043,0.20504},
  {0.74617,0.95593,0.20406},
  {0.75617,0.95121,0.20343},
  {0.76608,0.94627,0.20311},
  {0.77591,0.94113,0.20310},
  {0.78563,0.93579,0.20336},
  {0.79524,0.93025,0.20386},
  {0.80473,0.92452,0.20459},
  {0.81410,0.91861,0.20552},
  {0.82333,0.91253,0.20663},
  {0.83241,0.90627,0.20788},
  {0.84133,0.89986,0.20926},
  {0.85010,0.89328,0.21074},
  {0.85868,0.88655,0.21230},
  {0.86709,0.87968,0.21391},
  {0.87530,0.87267,0.21555},
  {0.88331,0.86553,0.21719},
  {0.89112,0.85826,0.21880},
  {0.89870,0.85087,0.22038},
  {0.90605,0.84337,0.22188},
  {0.91317,0.83576,0.22328},
  {0.92004,0.82806,0.22456},
  {0.92666,0.82025,0.22570},
  {0.93301,0.81236,0.22667},
  {0.93909,0.80439,0.22744},
  {0.94489,0.79634,0.22800},
  {0.95039,0.78823,0.22831},
  {0.95560,0.78005,0.22836},
  {0.96049,0.77181,0.22811},
  {0.96507,0.76352,0.22754},
  {0.96931,0.75519,0.22663},
  {0.97323,0.74682,0.22536},
  {0.97679,0.73842,0.22369},
  {0.98000,0.73000,0.22161},
  {0.98289,0.72140,0.21918},
  {0.98549,0.71250,0.21650},
  {0.98781,0.70330,0.21358},
  {0.98986,0.69382,0.21043},
  {0.99163,0.68408,0.20706},
  {0.99314,0.67408,0.20348},
  {0.99438,0.66386,0.19971},
  {0.99535,0.65341,0.19577},
  {0.99607,0.64277,0.19165},
  {0.99654,0.63193,0.18738},
  {0.99675,0.62093,0.18297},
  {0.99672,0.60977,0.17842},
  {0.99644,0.59846,0.17376},
  {0.99593,0.58703,0.16899},
  {0.99517,0.57549,0.16412},
  {0.99419,0.56386,0.15918},
  {0.99297,0.55214,0.15417},
  {0.99153,0.54036,0.14910},
  {0.98987,0.52854,0.14398},
  {0.98799,0.51667,0.13883},
  {0.98590,0.50479,0.13367},
  {0.98360,0.49291,0.12849},
  {0.98108,0.48104,0.12332},
  {0.97837,0.46920,0.11817},
  {0.97545,0.45740,0.11305},
  {0.97234,0.44565,0.10797},
  {0.96904,0.43399,0.10294},
  {0.96555,0.42241,0.09798},
  {0.96187,0.41093,0.09310},
  {0.95801,0.39958,0.08831},
  {0.95398,0.38836,0.08362},
  {0.94977,0.37729,0.07905},
  {0.94538,0.36638,0.07461},
  {0.94084,0.35566,0.07031},
  {0.93612,0.34513,0.06616},
  {0.93125,0.33482,0.06218},
  {0.92623,0.32473,0.05837},
  {0.92105,0.31489,0.05475},
  {0.91572,0.30530,0.05134},
  {0.91024,0.29599,0.04814},
  {0.90463,0.28696,0.04516},
  {0.89888,0.27824,0.04243},
  {0.89298,0.26981,0.03993},
  {0.88691,0.26152,0.03753},
  {0.88066,0.25334,0.03521},
  {0.87422,0.24526,0.03297},
  {0.86760,0.23730,0.03082},
  {0.86079,0.22945,0.02875},
  {0.85380,0.22170,0.02677},
  {0.84662,0.21407,0.02487},
  {0.83926,0.20654,0.02305},
  {0.83172,0.19912,0.02131},
  {0.82399,0.19182,0.01966},
  {0.81608,0.18462,0.01809},
  {0.80799,0.17753,0.01660},
  {0.79971,0.17055,0.01520},
  {0.79125,0.16368,0.01387},
  {0.78260,0.15693,0.01264},
  {0.77377,0.15028,0.01148},
  {0.76476,0.14374,0.01041},
  {0.75556,0.13731,0.00942},
  {0.74617,0.13098,0.00851},
  {0.73661,0.12477,0.00769},
  {0.72686,0.11867,0.00695},
  {0.71692,0.11268,0.00629},
  {0.70680,0.10680,0.00571},
  {0.69650,0.10102,0.00522},
  {0.68602,0.09536,0.00481},
  {0.67535,0.08980,0.00449},
  {0.66449,0.08436,0.00424},
  {0.65345,0.07902,0.00408},
  {0.64223,0.07380,0.00401},
  {0.63082,0.06868,0.00401},
  {0.61923,0.06367,0.00410},
  {0.60746,0.05878,0.00427},
  {0.59550,0.05399,0.00453},
  {0.58336,0.04931,0.00486},
  {0.57103,0.04474,0.00529},
  {0.55852,0.04028,0.00579},
  {0.54583,0.03593,0.00638},
  {0.53295,0.03169,0.00705},
  {0.51989,0.02756,0.00780},
  {0.50664,0.02354,0.00863},
  {0.49321,0.01963,0.00955},
  {0.47960,0.01583,0.01055}
};
constexpr double alpha_scaling_factor = 1e35;
constexpr double unit_length_in_meters = 1.616255e-35 * alpha_scaling_factor;
constexpr double unit_charge_in_electron_charges = 11.71;
constexpr double unit_time_in_seconds = 5.391247e-44 * alpha_scaling_factor;
constexpr double electron_mass_in_kg = 9.1093837015e-31;
constexpr double unit_mass_in_kg = 2.176434e-8 / alpha_scaling_factor;

constexpr double kg_in_unit_masses = 1.0 / unit_mass_in_kg;
constexpr double meter_in_unit_lengths = 1.0 / unit_length_in_meters;
constexpr double electron_charge_in_unit_charges = 1.0 / unit_charge_in_electron_charges;
constexpr double second_in_unit_times = 1.0 / unit_time_in_seconds;

constexpr double electron_mass_in_unit_masses = electron_mass_in_kg * kg_in_unit_masses;

uint64_t nanoTime(){
    using namespace std;
    using namespace chrono;
    return duration_cast<nanoseconds>(high_resolution_clock::now().time_since_epoch()).count();
}
template<typename T, unsigned N>
constexpr T pow(T x){
    T ret = 1.0;
    for(unsigned n = 0;n < N;n++){
        ret *= x;
    }
    return ret;
}
template<typename T>
T find_root(auto&& func, T start){
    T value = start;
    T last_error = 1e10;
    for(int i = 0;i < 1000;i++){
        jet<T, 1> tschet(value, T(1));
        jet<T, 1> eval = func(tschet);
        value -= eval.value / eval.deriv[0];
    }
    if(abs(func(jet<T, 1>(value, 0)).value) > 1e-3){
        std::cerr << "Did not converge: " << abs(func(jet<T, 1>(value, 0)).value) << "\n";   
    }
    return value;
}
template<typename T>
struct is_vector{
    constexpr static bool value = false;
};
template<typename T>
struct is_vector<std::vector<T>>{
    constexpr static bool value = true;
};
template<typename T>
constexpr bool is_vector_v = is_vector<T>::value;
template<typename T, typename callable>
    requires(!is_vector_v<callable>)
T find_tr(Vector<T, 3> pos, T t, callable traj){
    T start = t - (pos - traj(t)).norm();
    T ret = find_root([&](auto x){return jet<T, 1>(t, 0.0) - (pos.template cast<jet<T, 1>>() - traj(x)).norm() - x;}, start);
    assert(std::abs((t - ret) - (pos - traj(ret)).norm()) < 1e-13);
    return ret;
}
template <typename T, typename itype = size_t>
struct tret_hit{
    T value;
    itype lower;
    itype upper;
    //template<typename O, typename OI>
    //tret_hit(tret_hit<O, OI> o) : value(o.value), lower(o.lower), upper(o.upper){}
};
template<typename T>
struct traj_point{
    Vector<T, 3> pos;
    Vector<T, 3> beta;
};
template<typename T>
tret_hit<T> find_tr(Vector<T, 3> pos, T t, const std::vector<std::pair<T, traj_point<T>>>& traj){
    size_t beg = 0;
    size_t end = traj.size() - 1;
    std::ranges::for_each(traj, [&](auto x){
        //std::cout << t - x.first - (pos - x.second).norm() << "\n";
    });
    auto retard_equation = [&](size_t i){
        return t - traj[i].first - (pos - traj[i].second.pos).norm();
    };
    if(t - traj[beg+1].first - (pos - traj[beg+1].second.pos).norm() <= 0){
        //std::cerr << "klonk\n";
        //std::cerr << retard_equation(0) << " and " << retard_equation(traj.size() - 1) << "\n";
        return {0,(size_t)-1, (size_t)-1};
    }
    if(t - traj[end-1].first - (pos - traj[end-1].second.pos).norm() >= 0){
        std::cerr << "plonk\n";
        std::cerr << retard_equation(0) << " and " << retard_equation(traj.size() - 1) << " and " << traj[end-1].first << "\n";
        return {0,(size_t)-1, (size_t)-1};
    }
    do{
        assert(t - traj[beg].first - (pos - traj[beg].second.pos).norm() >= 0);
        assert(t - traj[end].first - (pos - traj[end].second.pos).norm() <= 0);
        size_t mp = (beg + end) / 2;
        Vector<T, 3> bv = traj[beg].second.pos;
        Vector<T, 3> ev = traj[end].second.pos;
        Vector<T, 3> mv = traj[mp] .second.pos;
        T btime = traj[beg].first;
        T etime = traj[end].first;
        T mtime = traj[mp] .first;

        T btime_eval = t - btime - (pos - traj[beg].second.pos).norm();
        T etime_eval = t - etime - (pos - traj[end].second.pos).norm();
        T mtime_eval = t - mtime - (pos - traj[mp] .second.pos).norm();
        if(mtime_eval > 0){
            beg = mp;
        }
        else if(mtime_eval < 0){
            end = mp;
        }
        else{
            return {btime, mp};
        }
    }while(end - beg > 3);
    for(size_t i = beg+1;i <= end;i++){
        if(retard_equation(i-1) >= 0 && retard_equation(i) <= 0){
            T w1 = abs(retard_equation(i-1));
            T w2 = abs(retard_equation(i));
            return {(traj[i - 1].first * w1 + traj[i].first * w2) / (w1 + w2), i - 1, i};
        }
    }
    assert(false);
    return {0, (size_t)-1};
    //T start = t - (pos - traj(t)).norm();
    //T ret = find_root([&](auto x){return jet<T, 1>(t, 0.0) - (pos.template cast<jet<T, 1>>() - traj(x)).norm() - x;}, start);
    //assert(std::abs((t - ret) - (pos - traj(ret)).norm()) < 1e-13);
    //return ret;
}
template<typename T>
T retarded_phi(Vector<T, 3> pos, T t, const std::vector<std::pair<T, Vector<T, 3>>>& traj){
    auto [t_r, index] = find_tr(pos, t, traj);
    Vector<T, 3> r_s = traj[index].second;
    Vector<T, 3> beta = (traj[index + 1].second - traj[index - 1].second) / (traj[index + 1].first - traj[index - 1].first);

    //std::cerr << "Beta = " << beta.transpose().norm() << "\n";
    Vector<T, 3> n_s = (pos - r_s).normalized();    

    return (1.0 / (4 * M_PI)) * 1.0 / ((1.0 - n_s.dot(beta)) * (pos - r_s).norm());
}
template<typename T, typename callable>
    requires(!is_vector_v<callable>)
T retarded_phi(Vector<T, 3> pos, T t, callable traj){
    T t_r = find_tr(pos, t, traj);
    Vector<T, 3> r_s = traj(t_r);    
    Vector<jet<T, 1>, 3> r_sjet_ = traj(jet<T, 1>(t_r, 1.0));
    Vector<T, 3> beta;

    for(int i = 0;i < 3;i++){
        beta(i) = r_sjet_(i).deriv[0];
    }
    //std::cerr << "Beta = " << beta.transpose().norm() << "\n";
    Vector<T, 3> n_s = (pos - r_s).normalized();    

    return (1.0 / (4 * M_PI)) * 1.0 / ((1.0 - n_s.dot(beta)) * (pos - r_s).norm());
}
template<typename T>
Vector<T, 3> retarded_A(Vector<T, 3> pos, T t, auto&& traj){
    T phi = retarded_phi(pos, t, traj);
    T t_r = find_tr(pos, t, traj);

    Vector<jet<T, 1>, 3> r_sjet_ = traj(jet<T, 1>(t_r, 1.0));
    Vector<T, 3> beta;
    for(int i = 0;i < 3;i++){
        beta(i) = r_sjet_(i).deriv[0];
    }
    return Vector<T, 3>(beta * phi);
}
template<typename T, typename callable, typename callable2>
    requires(!is_vector_v<callable> && !is_vector_v<callable2>)
Vector<T, 3> retarded_E(Vector<T, 3> pos, T t, callable traj, callable2 traj_bd){
    T t_r = find_tr(pos, t, traj);
    Vector<T, 3> r_s = traj(t_r);    
    Vector<jet<T, 1>, 3> r_sjet_ = traj(jet<T, 1>(t_r, 1.0));   
    Vector<T, 3> beta;

    for(int i = 0;i < 3;i++){
        beta(i) = r_sjet_(i).deriv[0];
    }
    T gamma = 1.0 / std::sqrt(1 - beta.dot(beta));
    Vector<T, 3> n_s = (pos - r_s).normalized();

    double q = 1.0;
    double c = 1.0;
    Vector<T, 3> first_part = q * (n_s - beta) / (gamma * gamma * pow<T, 3>(1.0 - n_s.dot(beta)) * (pos - r_s).squaredNorm());
    Vector<T, 3> second_part = (q * n_s).cross((n_s - beta).cross(traj_bd(t_r))) / (c * pow<T, 3>(1.0 - n_s.dot(beta)) * (pos - r_s).norm());
    //Vector<T, 3> second_part = (q * n_s).cross((n_s - beta).cross(traj_bd(t_r)));
    //std::cout << "Secon: " << second_part << "\n";
    return (1.0 / (4 * M_PI)) * (first_part + second_part);
    
}
template<typename T>
Vector<T, 3> retarded_E(const Vector<T, 3>& pos, const Vector<T, 3>& pos_at_retard, const Vector<T, 3>& ret_beta, const Vector<T, 3>& ret_accel, T ret_time){
    double q = 1.0;
    double c = 1.0;
    Vector<T, 3> n_s = (pos - pos_at_retard).normalized();
    Vector<T, 3> second_part = (q * n_s).cross((n_s - ret_beta).cross(ret_accel)) / (c * pow<T, 3>(1.0 - n_s.dot(ret_beta)) * (pos - pos_at_retard).norm());
    return (1.0 / (4 * M_PI)) * second_part;
}
template<typename T>
Vector<T, 3> retarded_E(Vector<T, 3> pos, T t, const std::vector<std::pair<T, traj_point<T>>>& traj){
    auto [t_r, index_before, index_after] = find_tr(pos, t, traj);
    if(index_before < 1 || index_before >= traj.size() - 1){
        //std::cerr << "the fuck???: " << t << " at " << pos << "\n";
        return Vector<T, 3>{0,0,0};
    }
    Vector<T, 3> pos_at_retard = traj[index_before].second.pos;
    Vector<T, 3> beta = traj[index_before].second.beta;
    Vector<T, 3> accel = (traj[index_before + 1].second.beta - traj[index_before - 1].second.beta) / (traj[index_before + 1].first - traj[index_before - 1].first);

    //Vector<T, 3> accel = ((traj[index_before + 1].second - traj[index_before].second) / (traj[index_before + 1].first - traj[index_before].first)
    //                   - (traj[index_before].second - traj[index_before - 1].second) / (traj[index_before].first - traj[index_before - 1].first)) / (0.5 * (traj[index_before + 1].first - traj[index_before - 1].first));

    
    T gamma = 1.0 / std::sqrt(1 - beta.dot(beta));
    Vector<T, 3> n_s = (pos - pos_at_retard).normalized();

    double q = 1.0;
    double c = 1.0;
    Vector<T, 3> first_part = q * (n_s - beta) / (gamma * gamma * pow<T, 3>(1.0 - n_s.dot(beta)) * (pos - pos_at_retard).squaredNorm());
    Vector<T, 3> second_part = (q * n_s).cross((n_s - beta).cross(accel)) / (c * pow<T, 3>(1.0 - n_s.dot(beta)) * (pos - pos_at_retard).norm());
    //Vector<T, 3> second_part = (q * n_s).cross((n_s - beta).cross(traj_bd(t_r)));
    //std::cout << "Secon: " << second_part << "\n";
    return (1.0 / (4 * M_PI)) * second_part;
}

template<typename T, typename callable, typename callable2>
    requires(!is_vector_v<callable> && !is_vector_v<callable2>)
Vector<T, 3> retarded_B(Vector<T, 3> pos, T t, callable traj, callable2 traj_bd){
    //assert(false);
    
    T t_r = find_tr(pos, t, traj);
    Vector<T, 3> r_s = traj(t_r);    
    Vector<jet<T, 1>, 3> r_sjet_ = traj(jet<T, 1>(t_r, 1.0));   
    Vector<T, 3> beta;

    for(int i = 0;i < 3;i++){
        beta(i) = r_sjet_(i).deriv[0];
    }
    T gamma = 1.0 / std::sqrt(1 - beta.dot(beta));
    Vector<T, 3> n_s = (pos - r_s).normalized();

    Vector<T, 3> E = retarded_E(pos, t, traj, traj_bd);
    Vector<T, 3> ret = n_s.cross(E);
    return ret;
}
template<typename T>
Vector<T, 3> retarded_B(Vector<T, 3> pos, T t, const std::vector<std::pair<T, traj_point<T>>>& traj){
    auto [t_r, index_before, index_after] = find_tr(pos, t, traj);
    if(index_before < 1 || index_after >= traj.size() - 1){
        return Vector<T, 3>{0,0,0};
    }
    Vector<T, 3> pos_at_retard = traj[index_before].second.pos;
    Vector<T, 3> beta = traj[index_before].second.beta;
    Vector<T, 3> accel = (traj[index_before + 1].second.beta - traj[index_before - 1].second.beta) / (traj[index_before + 1].first - traj[index_before - 1].first);

    //Vector<T, 3> accel = ((traj[index_before + 1].second - traj[index_before].second) / (traj[index_before + 1].first - traj[index_before].first)
    //                   - (traj[index_before].second - traj[index_before - 1].second) / (traj[index_before].first - traj[index_before - 1].first)) / (0.5 * (traj[index_before + 1].first - traj[index_before - 1].first));
    //std::cerr << beta.transpose() << "\n";
    //Vector<T, 3> beta = (traj[index + 1].second - traj[index - 1].second) / (traj[index + 1].first - traj[index - 1].first);
    T gamma = 1.0 / std::sqrt(1 - beta.squaredNorm());
    Vector<T, 3> n_s = (pos - pos_at_retard).normalized();
    Vector<T, 3> E = retarded_E(pos, t, traj);
    return n_s.cross(E);
    double q = 1.0;
    double c = 1.0;
    Vector<T, 3> first_part = q * c * (beta.cross(n_s)) / (gamma * gamma * pow<T, 3>(1.0 - n_s.dot(beta)) * (pos - pos_at_retard).squaredNorm());
    Vector<T, 3> second_part = (q * n_s).cross(n_s.cross((n_s - beta).cross(accel))) / (pow<T, 3>(1.0 - n_s.dot(beta)) * (pos - pos_at_retard).norm());
    return (1.0 / (4 * M_PI)) * (first_part * 0 + second_part);
}
template<typename T, typename callable, typename callable2>
    requires(!is_vector_v<callable> && !is_vector_v<callable2>)
Vector<T, 3> retarded_radiation(Vector<T, 3> pos, T t, callable traj, callable2 traj_bd){
    return retarded_E(pos, t, traj, traj_bd).cross(retarded_B(pos, t, traj, traj_bd));
}
template<typename T>
Vector<T, 3> retarded_radiation(Vector<T, 3> pos, T t, const std::vector<std::pair<T, traj_point<T>>>& traj){
    auto [t_r, index_before, index_after] = find_tr(pos, t, traj);
    if(index_before < 1 || index_after >= traj.size() - 1 || index_before == (size_t(-1))){
        return Vector<T, 3>{0,0,0};
    }
    Vector<T, 3> pos_at_retard = traj[index_before].second.pos;
    Vector<T, 3> beta = traj[index_before].second.beta;
    Vector<T, 3> beta_next = traj[index_after].second.beta;
    Vector<T, 3> accel = (traj[index_before + 1].second.beta - traj[index_before - 1].second.beta) / (traj[index_before + 1].first - traj[index_before - 1].first);
    Vector<T, 3> n_s = (pos - pos_at_retard).normalized();
    Vector<T, 3> ret = retarded_E(pos, t, traj).cross(n_s.cross(retarded_E(pos, t, traj)));
    //if(ret[1] == INFINITY){
    //    std::cerr << n_s << "\n";
    //}
    //std::cerr << ret.transpose() << "\n";
    return ret;
    /*Vector<T, 3> ret(0,0,0);
    Eigen::Quaternion<T> qa;
    qa.setIdentity();
    Eigen::Quaternion<T> qb;
    qb.setFromTwoVectors(beta, beta_next);
    for(size_t i = 0;i < 1;i++){
        double x = double(i) / 1.0;
        Vector<T, 3> cbeta = qa.slerp(t, qb) * beta;
        ret += retarded_E(pos, pos_at_retard, cbeta, accel, t_r).cross(n_s.cross(retarded_E(pos, pos_at_retard, cbeta, accel, t_r)));
    }
    return ret / 1.0;*/
}
void pgrid(auto&& func){
    for(double x = 0;x <= 1000;x += 0.125){
        std::cout << "func(" << x << ") = " << func(x).value << "\n";
    }
}
constexpr double omega  = 4.5;
constexpr double radius = 0.2;

auto traj(auto t){
    using T = decltype(t);
    return Eigen::Matrix<T, 3, 1>{radius * cos(t * omega), radius * sin(t * omega), 0};
}
Vector<double, 3> traj_bd(double t){
    using T = double;
    return Eigen::Matrix<T, 3, 1>(radius * omega * omega * Eigen::Matrix<T, 3, 1>{-cos(t * omega), -sin(t * omega), 0});
}
template<typename T>
struct sample_point{
    Vector<T, 3> pos;
    Vector<T, 3> normal;
};

template<typename T, typename gentype>
sample_point<T> random_point_on_unit_sphere(gentype& gen){
    using std::sin;
    using std::cos;
    using std::acos;
    std::uniform_real_distribution<T> dis(0, T(1.0));
    std::uniform_real_distribution<T> disp(0, 2 * M_PI);

    const T u = disp(gen);
    const T v = dis(gen);
    T phi     = u;
    T theta   = acos(2 * v - 1);
    Vector<T, 3> offs{cos(phi) * sin(theta), sin(phi) * sin(theta),
                                              cos(theta)};
    sample_point<T> ret;
    ret.pos = offs;
    ret.normal = offs;
    return ret;
}

constexpr double measurement_radius = 1.0;
template<typename T, typename gentype>
std::vector<sample_point<T>> generate_samplepoints(gentype& gen, size_t n){
    using std::sin;
    using std::cos;
    using std::acos;
    std::uniform_real_distribution<T> dis(0, T(1.0));
    std::uniform_real_distribution<T> disp(0, 2 * M_PI);
    size_t mod = std::ceil(std::sqrt(T(n)));
    n = mod * mod;
    std::vector<sample_point<T>> ret(n);
    for(size_t i = 0;i < n;i++){

        const T u = 2.0 * M_PI * T(i / mod) / T(mod);
        const T v = T(i % mod) / T(mod);
        T phi     = u;
        T theta   = acos(2 * v - 1);
        Vector<T, 3> offs{cos(phi) * sin(theta), sin(phi) * sin(theta),
                                                  cos(theta)};
        sample_point<T> ret_p;
        ret_p.pos = offs * measurement_radius;
        ret_p.normal = offs.normalized();
        ret[i] = ret_p;
    }
    return ret;
}
namespace ippl{
    template<typename T, unsigned N>
    using Vector = Eigen::Matrix<T, N, 1>;
    template<typename T, typename U>
    using pear = std::pair<T, U>;
}
template<typename scalar>
struct static_undulator{
    scalar k_u;
    scalar undulator_parameter;
    scalar B_magnitude;
    scalar length;
    scalar distance_to_entry;
    constexpr static_undulator(scalar lambda_u_, scalar undulator_parameter_, scalar length_, scalar distance_to_entry_)
     : k_u((2.0 * M_PI) / lambda_u_), undulator_parameter(undulator_parameter_), B_magnitude((2 * M_PI * electron_mass_in_unit_masses * undulator_parameter) / (electron_charge_in_unit_charges * lambda_u_)), length(length_), distance_to_entry(distance_to_entry_){
        
    }
    constexpr ippl::pear<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> operator()(const ippl::Vector<scalar, 3>& position_in_lab_frame)const noexcept{
        ippl::pear<ippl::Vector<scalar, 3>, ippl::Vector<scalar, 3>> ret;
        ret.first.fill(0);
        if(position_in_lab_frame[2] < distance_to_entry){
            //assert(false);
            scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
            assert(z_in_undulator < 0);
            scalar scal = exp(-((k_u * z_in_undulator) * (k_u * z_in_undulator) * 0.5));
            //printf("Discard: %.4e\n", scal);
            ret.second[0] = 0;
            ret.second[1] = B_magnitude * cosh(k_u * position_in_lab_frame[1]) * z_in_undulator * k_u * scal;
            ret.second[2] = B_magnitude * sinh(k_u * position_in_lab_frame[1]) * scal;
        }
        else if(position_in_lab_frame[2] > distance_to_entry && position_in_lab_frame[2] < distance_to_entry + length){
            scalar z_in_undulator = position_in_lab_frame[2] - distance_to_entry;
            assert(z_in_undulator >= 0);
            ret.second[0] = 0;
            ret.second[1] = B_magnitude * cosh(k_u * position_in_lab_frame[1]) * sin(k_u * z_in_undulator);
            ret.second[2] = B_magnitude * sinh(k_u * position_in_lab_frame[1]) * cos(k_u * z_in_undulator);
        }
        else{
            ret.second.fill(0);
        }
        return ret;
        
    }
};
template<typename T>
constexpr double gamma_to_beta(T gamma){
    using std::sqrt;
    return sqrt(gamma * gamma - 1) / gamma;
}
template<typename T>
constexpr double beta_to_gamma(T beta){
    using std::sqrt;
    return 1.0 / sqrt(1 - beta * beta);
}
template<typename T>
constexpr double gammabeta_to_gamma(T gammabeta){
    using std::sqrt;
    return sqrt(1 + gammabeta * gammabeta);
}
template<typename T>
Eigen::Matrix<T, 3, 1> gammabeta_to_beta(Eigen::Matrix<T, 3, 1> gammabeta){
    using std::sqrt;
    return gammabeta / sqrt(1.0 + gammabeta.squaredNorm());
}
template<typename scalar>
using trajectory = std::vector<std::pair<scalar, traj_point<scalar>>>;
template<typename scalar>
std::vector<std::pair<scalar, traj_point<scalar>>> do_run(double gamma, double K, bool verbose = false){
    using Vector3 = Eigen::Matrix<scalar, 3, 1>;
    std::vector<std::pair<scalar, traj_point<scalar>>> trajectory;
    Vector3 pos{0.0, 1000.0 * 1e-6 * meter_in_unit_lengths, 0.0};
    Vector3 gammabeta{0, 0, gamma_to_beta(gamma) * gamma};
    const scalar charge = electron_charge_in_unit_charges;
    const scalar mass = electron_mass_in_unit_masses;
    const scalar lambda_u = 3e4 * (1e-6) * meter_in_unit_lengths;
    const scalar length = 5.0 * meter_in_unit_lengths;
    const scalar bunch_dt = length / (2.0 * (gamma * gamma + 100 * gamma + 50000));
    const scalar k_u = 2 * M_PI / lambda_u;
    const scalar distance_to_entry = 5.0 * lambda_u;
    const scalar B_magnitude = (2 * M_PI * electron_mass_in_unit_masses * K) / (electron_charge_in_unit_charges * lambda_u);
    static_undulator<scalar> undulator_field(lambda_u, K, length, distance_to_entry);
    uint64_t step = 0;
    for(;;step++){
        using std::sqrt;
        const ippl::Vector<scalar, 3> pgammabeta = gammabeta;
        std::pair<Vector3, Vector3> ufield = undulator_field(pos);
        //std::pair<Vector3d, Vector3d> ufield{Vector3d(0,0,0), Vector3d(0,0,0)};
        Vector3 E = ufield.first;
        Vector3 B = ufield.second;
        
        const ippl::Vector<scalar, 3> t1 = pgammabeta - charge * bunch_dt * E / (scalar(2) * mass);
        const scalar alpha = -charge * bunch_dt / (scalar(2) * mass * sqrt(1 + t1.dot(t1)));
        const ippl::Vector<scalar, 3> t2 = t1 + alpha * t1.cross(B);
        const ippl::Vector<scalar, 3> t3 = t1 + t2.cross(scalar(2) * alpha * (B / (1.0 + alpha * alpha * (B.dot(B)))));
        const ippl::Vector<scalar, 3> ngammabeta = t3 - charge * bunch_dt * E / (scalar(2) * mass);
        const ippl::Vector<scalar, 3> nbeta = ngammabeta * (scalar(1.0) / sqrt(1 + ngammabeta.dot(ngammabeta)));
        assert(nbeta.squaredNorm() < 1);
        pos = pos + bunch_dt * nbeta;
        trajectory.emplace_back((step + 1) * bunch_dt, traj_point<scalar>{.pos = pos, .beta = nbeta});
        //std::cout << "Time: " << (step + 1) * bunch_dt << "\n";
        //std::cout << "Z: " << pos.z() << "\n";
        if(pos.z() >= length + distance_to_entry){
            break;
        }
    }
    uint64_t next = step + 1000;
    for(;step < next;step++){
        using std::sqrt;
        const ippl::Vector<scalar, 3> pgammabeta = gammabeta;
        std::pair<Vector3, Vector3> ufield = undulator_field(pos);
        //std::pair<Vector3d, Vector3d> ufield{Vector3d(0,0,0), Vector3d(0,0,0)};
        Vector3 E = ufield.first;
        Vector3 B = ufield.second;
        
        const ippl::Vector<scalar, 3> t1 = pgammabeta - charge * bunch_dt * E / (scalar(2) * mass);
        const scalar alpha = -charge * bunch_dt / (scalar(2) * mass * sqrt(1 + t1.dot(t1)));
        const ippl::Vector<scalar, 3> t2 = t1 + alpha * t1.cross(B);
        const ippl::Vector<scalar, 3> t3 = t1 + t2.cross(scalar(2) * alpha * (B / (1.0 + alpha * alpha * (B.dot(B)))));
        const ippl::Vector<scalar, 3> ngammabeta = t3 - charge * bunch_dt * E / (scalar(2) * mass);
        const ippl::Vector<scalar, 3> nbeta = ngammabeta * (scalar(1.0) / sqrt(1 + ngammabeta.dot(ngammabeta)));
        assert(nbeta.squaredNorm() < 1);
        pos = pos + bunch_dt * nbeta;
        trajectory.emplace_back((step + 1) * bunch_dt, traj_point<scalar>{.pos = pos, .beta = nbeta});
        //std::cout << "Time: " << (step + 1) * bunch_dt << "\n";
        //std::cout << "Z: " << pos.z() << "\n";
    }
    return trajectory;
}
template<typename scalar>
trajectory<scalar> dipole_trajectory(const scalar omega, const size_t steps, const scalar ttime){
    using vec3 = Eigen::Vector3<scalar>;
    scalar iomega = scalar(1) / omega;
    using std::sin;
    using std::cos;
    trajectory<scalar> ret(steps);
    for(size_t i = 0;i < steps;i++){
        scalar time = scalar(i) * ttime / scalar(steps);
        ret[i].first = time;
        ret[i].second.pos = vec3{0.99 * iomega * sin(omega * time), 0, 0.99 * iomega * cos(omega * time)};
        ret[i].second.beta = vec3{0.99 * cos(omega * time), 0, 0.99 * -sin(omega * time)};
    }
    return ret;
}
template<typename scalar>
trajectory<scalar> boosted_dipole(const scalar gamma, const scalar omega, const size_t steps, const scalar ttime){
    using vec3 = Eigen::Vector3<scalar>;
    scalar real_omega = omega / gamma;
    scalar iomega = scalar(1) / real_omega;
    scalar beta = gamma == 1.0 ? 0 : std::sqrt(1 - 1.0 / (double(gamma) * double(gamma)));
    using std::sin;
    using std::cos;
    trajectory<scalar> ret(steps);
    for(size_t i = 0;i < steps;i++){
        scalar time = scalar(i) * ttime / scalar(steps);
        ret[i].first = time;
        ret[i].second.pos = vec3{iomega * sin(real_omega * time), 0, beta * time};
        ret[i].second.beta = vec3{cos(real_omega * time), 0, beta};
    }
    return ret;
}
template<typename... Args>
std::string form(const char* fmt, Args&&... args){
    char buf[2048] = {0};
    std::snprintf(buf, 2048, fmt, std::forward<Args>(args)...);
    return std::string(buf);
}
int mainframebunch(){
    using scalar = double;
    double gamma = 3.0;
    double beta = std::sqrt(1 - 1.0 / gamma);
    double time = 500.0;
    auto tr0 = boosted_dipole(1.0, 0.9, (1 << 15), time / gamma);
    auto tr1 = boosted_dipole(gamma, 0.9, (1 << 15), time);
    size_t scount = 100000;
    std::ofstream traj("traj.txt");
    for(size_t i = 0;i < tr1.size();i++){
        traj << tr0[i].second.pos.transpose() << "\n";
    }
    std::ofstream zradb("zrad_bunch.txt");
    for(double t = 0.0;t <= 200.0;t += 0.01){
        double rad = retarded_radiation(Vector<double, 3>{0, 0, 1}, t, tr0).z();
        /*#pragma omp parallel for schedule(guided) reduction(+:rad)
        for(size_t s = 0;s < scount;s++){
            double v = double(s) / scount;
            Vector<double, 3> spoint{25 * std::cos(v),25 * std::sin(v),0.0};
            Vector<double, 3> snormal{std::cos(v),std::sin(v),0.0};
            rad += snormal.dot(retarded_radiation(spoint, t, tr0));
        }*/
        zradb << t << " " << rad << "\n";
    }
    return 0;
    std::ofstream zradl("zrad_lab.txt");
    for(double t = 180;t <= 220.0;t += 0.01){
        double rad = retarded_radiation({0, 0, 200}, t, tr1).z();
        zradl << t << " " << rad << "\n";
    }
    /*
    for(double t = 0.0;t <= 400.0;t += 0.01){
        double rad = 0;
        #pragma omp parallel for schedule(guided) reduction(+:rad)
        for(size_t s = 0;s < scount;s++){
            double v = double(s) / scount;
            Vector<double, 3> spoint{250.0 * std::cos(v),250.0 * std::sin(v),0.0};
            Vector<double, 3> snormal{std::cos(v), std::sin(v), 0.0};
            rad += snormal.dot(retarded_radiation(spoint, t, tr1));
        }
        zradl << t << " " << rad / 100.0 << "\n";
    }*/
    return 0;
}
int main(){
    using scalar = double;
    const double gamma = 10.0;
    const scalar rad = 0.2;
    const scalar omega = std::sqrt(1 - 1.0 / (gamma * gamma)) / rad;
    auto trajectory = [=](auto t){
        return Vector<decltype(t), 3>{scalar(cos(t * omega) * rad), scalar(sin(t * omega) * rad), (scalar)0.0};
    };
    auto trajectory_d = [=](auto t){
        return Vector<decltype(t), 3>{scalar(-omega * sin(t * omega) * rad), scalar(omega * cos(t * omega) * rad), (scalar)0.0};
    };
    
    std::vector<std::pair<scalar, traj_point<scalar>>> traj_explicit;// = do_run<scalar>(gamma, 100.0);
    for(double t = -10;t <= 10;t += 1.0 / 65536){
        traj_explicit.emplace_back(t, traj_point<scalar>{.pos = trajectory(t), .beta = trajectory_d(t)});
    }
    //std::vector<std::pair<scalar, traj_point<scalar>>> traj_explicit = dipole_trajectory<scalar>(5.0, (1 << 22), 20.0);
    
    std::cerr << "Final time: " << traj_explicit.back().first << "\n";
    std::cerr << "Final pos: " << traj_explicit.back().second.pos << "\n";
    {
        std::ofstream traj("traj.txt");
        for(auto [t, tr] : traj_explicit){
            traj << tr.pos.transpose() << "\n";
        }
    }
    //return 0;
    //traj_explicit.resize(1 << 24);
    //for(size_t i = 0;i < traj_explicit.size();i++){
    //    scalar time = -scalar(5.0) + (scalar(5.0) * i) / (traj_explicit.size() - 1);
    //    traj_explicit[i] = {time, traj_point{trajectory(time), trajectory_d(time)}};
    //}
    Vector<scalar, 3> spoint = Vector<scalar, 3>{0,1.0,0};
    //double retard = find_tr(spoint, 0.0, traj_explicit);
    //std::cout << retard << "\n";
    //std::cout << find_tr(spoint, 0.0, trajectory) << "\n";
    //std::cout << (trajectory(find_tr(spoint, 0.0, trajectory)) - spoint).norm() << "\n";
    int m = 5000, n = 5000;
    
    Eigen::MatrixXd red(m, n);
    auto t1 = nanoTime();
    
    std::cout.precision(4);
    #pragma omp parallel for collapse(2) schedule(guided)
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;j++){
            //Vector<scalar, 3> spoint = Vector<scalar, 3>{scalar(2.3) * (scalar(i) / scalar(m) - scalar(0.5)), scalar(2.3) * (scalar(j) / scalar(n) - scalar(0.5)), 0};
            //Vector<scalar, 3> spoint = Vector<scalar, 3>{scalar(0.0075) * (scalar(i) / scalar(m) - scalar(0.5)), 0, 5 * meter_in_unit_lengths + 5 * 0.03 * meter_in_unit_lengths + scalar(0.015) * (scalar(j) / scalar(n) - scalar(0.5))};
            
            Vector<scalar, 3> spoint = Vector<scalar, 3>{2.0 * (scalar(i) / scalar(m) - scalar(0.5)), 2.0 * (scalar(j) / scalar(n) - scalar(0.5)), 0};
                
            //std::cout << "A";
            //red(i, j) = find_tr(spoint, (traj_explicit.end() - 1000)->first, traj_explicit).value;
            //std::cerr << find_tr(spoint, scalar(1.0), traj_explicit).upper << "\n";
            //red(i, j) = std::log(retarded_B(spoint, 0.0, traj_explicit).norm());
            scalar v = retarded_radiation(spoint, (traj_explicit.end() - 1000)->first, traj_explicit).norm();
            red(i, j) = v == 0 ? std::numeric_limits<scalar>::infinity() : std::log10(v);
        }
    }
    scalar minv = red.minCoeff();
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;j++){
            if(red(i,j) == std::numeric_limits<scalar>::infinity()){
                red(i,j) = minv;
            }
        }
    }
    std::vector<uint8_t> stbbuffer(m * n * 3);
    scalar zboost = -1;
    for(auto [t, p] : traj_explicit){
        if(t > 0.725){
            zboost = p.pos[2];
            break;
        }
    }
    std::cerr << "zboost: " << zboost << "\n";
    bool first = true;
    scalar idiff;
    for(scalar t = 0.0; t <= 4.0; t += 1.0 / 128.0){
        
        
        
        std::cerr << "T: " << t << "\n";
        #pragma omp parallel for collapse(2) schedule(guided)
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j++){
                Vector<scalar, 3> spoint = Vector<scalar, 3>{2.0 * (scalar(i) / scalar(m) - scalar(0.5)), 2.0 * (scalar(j) / scalar(n) - scalar(0.5)), 0};
                //Vector<scalar, 3> spoint = Vector<scalar, 3>{0.025 * (scalar(i) / scalar(m) - scalar(0.5)), 0, zboost + 0.1 * (scalar(j) / scalar(n) - scalar(0.5))};
                //red(i,j) = find_tr(spoint, t, traj_explicit).value;
                
                //std::cerr << find_tr(spoint, t, traj_explicit).value << "\n";
                scalar v = retarded_radiation(spoint, t, traj_explicit).norm();
                red(i, j) = v == 0 ? std::numeric_limits<scalar>::infinity() : std::log10(v);
            }
        }
        
        //std::cerr << red.sum() / (m * n) << "\n";
        if(first)
        {
            minv = red.minCoeff();
            std::vector<scalar> redv;
            for(int i = 0;i < m;i++){
                for(int j = 0;j < n;j++){
                    if(std::isinf(red(i,j)) || std::isnan(red(i,j))){
                        red(i, j) = minv;
                        
                    }
                    else{
                        redv.push_back(red(i, j));
                    }
                }
            }
            std::sort(redv.begin(), redv.end());
            minv = redv[0];
            scalar maxv = redv[redv.size() - 5];
            std::cerr << std::format("minv: {}\n", minv);
            std::cerr << std::format("maxv: {}\n", maxv);
            //minv = red.minCoeff();
            //if(std::isinf(minv)){
            //    std::abort();
            //}
            //scalar maxv = red.maxCoeff();
            idiff = 1.0 / (maxv - minv);
        }
        
        first = false;
        for(int i = 0;i < m;i++){
            for(int j = 0;j < n;j++){
                //std::cerr << "Idiff: " << minv << "\n";
                double inten = (red(i,j) - minv) * idiff;
                double accf = 255.0 * std::clamp(inten, 0.0, 1.0);
                double acci = 0;
                double frac = std::modf(accf, &acci);

                unsigned acc = acci;
                double r = turbo_cm[acc][0];
                double g = turbo_cm[acc][1];
                double b = turbo_cm[acc][2];
                if(acc < 255){
                    r = r * (1.0 - frac) + turbo_cm[acc + 1][0] * frac;
                    g = g * (1.0 - frac) + turbo_cm[acc + 1][1] * frac;
                    b = b * (1.0 - frac) + turbo_cm[acc + 1][2] * frac;
                }
                stbbuffer[(i * n + j) * 3 + 0] = uint8_t(r * 255);
                stbbuffer[(i * n + j) * 3 + 1] = uint8_t(g * 255);
                stbbuffer[(i * n + j) * 3 + 2] = uint8_t(b * 255);
            }
        }
        stbi_write_png(form("data/out%f.png", t).c_str(), n, m, 3, stbbuffer.data(), n * 3);
    }
    
    //std::cout << red << "\n";
    return 0;
    auto t2 = nanoTime();
    xoshiro_256 gen(12345);
    //std::vector<sample_point<scalar>> points = generate_samplepoints<double>(gen, 1 << 24);
    double total_radiation = 0;
    constexpr size_t pnroot = (1ull << 8);
    #pragma omp parallel for collapse(2) reduction(+: total_radiation)
    for(size_t xi = 0;xi < pnroot;xi++){
        for(size_t yi = 0;yi < pnroot;yi++){
            const scalar u = 2.0 * M_PI * scalar(xi) / scalar(pnroot);
            const scalar v = scalar(yi) / scalar(pnroot);
            const scalar phi     = u;
            const scalar theta   = acos(2 * v - 1);
            Vector<scalar, 3> offs{cos(phi) * sin(theta), sin(phi) * sin(theta),
                                                      cos(theta)};
            sample_point<scalar> ret_p;
            Vector<scalar, 3> pos = offs * measurement_radius;
            Vector<scalar, 3> normal = offs.normalized();
            double localaccum = 0.0;
            size_t cnt = 0;
            for(double t = -(2.0 * M_PI / double(pnroot) / omega);t <= 0.00;t += 1e-5){
                cnt++;
                localaccum += retarded_radiation(pos, t, traj_explicit).dot(normal);
            }
            localaccum *= 1.0 / cnt;
            total_radiation += localaccum;
            //total_radiation += retarded_radiation(points[i].pos, 0.0, traj_explicit).dot(points[i].normal);
        }
    }


    std::cerr << uint64_t((t2 - t1) / 1000) / 1000.0 << " ms\n";
    //std::cout << red << "\n";
    std::cerr.precision(15);
    std::cerr << "Radiation: " << (4 * M_PI * total_radiation * measurement_radius * measurement_radius) / pnroot / pnroot << "\n";
    //for(double t = -10;t < 10;t += 0.5){
    //    std::cerr << t << ": " << retarded_radiation(Vector<scalar, 3>{0,1,0}, t, traj_explicit).norm() << "\n";
    //}
    return 0;
}
/*int main2(int argc, char** argv){
    //jet<jet<double, 1>, 1> djet(jet<double>);
    jet<double, 1> tschet(1.0, 1);
    //std::cout << tschet << "\n" << traj(tschet).dot(traj(tschet)) << "\n\n";
    //std::cout << tschet << "\n" << traj(tschet).norm() << "\n\n";
    
    //return 0;
    using v3 = Eigen::Vector3d;
    Eigen::Matrix<double, 3, 1> r(0.9, 0.5, 0);
    //auto f = [r](auto x){return jet<double, 1>(10.0, 0.0) - (r.cast<decltype(x)>() - traj(x)).norm() - x;};
    //auto sq2 = find_root([r](auto x){return exp(x * x) - std::exp(4);}, 2.0);
    //pgrid([r](auto x){return jet<double, 1>(10.0, 0.0) - (r.cast<decltype(x)>() - traj(x)).norm() - x;});
    constexpr double a = 1e-10;
    xoshiro_256 gen(12345);
    std::vector<sample_point<double>> points = generate_samplepoints<double>(gen, 16384);
    //#pragma omp parallel for
    auto f   = [](auto x){return traj(x);};
    auto acc = [](auto x){return traj_bd(x);};

    const double upper_limit = 1.0 / omega;
    const double step = 1e-2 / omega;
    std::vector<std::atomic<double>> rad_results(upper_limit / step);
    for(auto& d : rad_results){d = 0;}

    #pragma omp parallel for collapse(2) schedule(guided)
    for(size_t istep = 0;istep < rad_results.size();istep++){
        //v3 rp = r;
        //v3 rm = r;
        //double tp = t + a;
        //double tm = t - a;
        //rp[0] += a; 
        //rm[0] -= a; 
        
        //auto phi   = retarded_phi(r, t, f);
        //auto Ex    = retarded_E(r, t, f, acc).x();
        //auto numex = (retarded_phi(rp, t, f) - retarded_phi(rm, t, f)) / (2 * a) + (retarded_A(r, tp, f).x() - retarded_A(r, tm, f).x()) / (2 * a);
        //std::cout << t << " " << phi << "\n";
        //double total_radiation = 0;
        //#pragma omp parallel for reduction(+: total_radiation)
        for(size_t i = 0;i < points.size();i++){
            //std::cerr << points[i].pos.transpose() << "\n";
            rad_results[istep] = rad_results[istep] + retarded_radiation(points[i].pos, step * istep, f, acc).dot(points[i].normal);
        }
        //std::cout << t << " " << total_radiation << "\n";
    }
    for(size_t istep = 0;istep < rad_results.size();istep++){
        rad_results[istep] = rad_results[istep] * (4.0 * M_PI / points.size()) * measurement_radius * measurement_radius;
        std::cout << istep * step << " " << rad_results[istep] << "\n";
    }
    //std::cout << sq2 << "\n";
    //std::cout << (double)f(sq2) << "\n";
    return 0;
}*/