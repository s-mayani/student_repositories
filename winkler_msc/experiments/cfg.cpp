#include "json.hpp"
#include "cfg.hpp"
#include <fstream>
#include <iostream>
#include <cassert>
template<typename scalar, unsigned Dim>
ippl::Vector<scalar, Dim> getVector(const nlohmann::json& j){
    if(j.is_array()){
        assert(j.size() == Dim);
        ippl::Vector<scalar, Dim> ret;
        for(unsigned i = 0;i < Dim;i++)
            ret[i] = (scalar)j[i];
        return ret;
    }
    else{
        std::cerr << "Warning: Obtaining Vector from scalar json\n";
        ippl::Vector<scalar, Dim> ret;
        ret.fill((scalar)j);
        return ret;
    }
}
template<size_t N, typename T>
struct DefaultedStringLiteral {
    constexpr DefaultedStringLiteral(const char (&str)[N], const T val) : value(val) {
        std::copy_n(str, N, key);
    }
    
    T value;
    char key[N];
};
template<size_t N>
struct StringLiteral {
    constexpr StringLiteral(const char (&str)[N]) {
        std::copy_n(str, N, value);
    }
    
    char value[N];
    constexpr DefaultedStringLiteral<N, int> operator>>(int t)const noexcept{
        return DefaultedStringLiteral<N, int>(value, t);
    }
    constexpr size_t size()const noexcept{return N - 1;}
};
template<StringLiteral lit>
constexpr size_t chash(){
    size_t hash = 5381;
    int c;

    for(size_t i = 0;i < lit.size();i++){
        c = lit.value[i];
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }

    return hash;
}
size_t chash(const char* val) {
    size_t hash = 5381;
    int c;

    while ((c = *val++)) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }

    return hash;
}
size_t chash(const std::string& _val) {
    size_t hash = 5381;
    const char* val = _val.c_str();
    int c;

    while ((c = *val++)) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }

    return hash;
}
std::string lowercase_singular(std::string str) {
    // Convert string to lowercase
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);

    // Check if the string ends with "s" and remove it if it does
    if (!str.empty() && str.back() == 's') {
        str.pop_back();
    }

    return str;
}
double get_time_multiplier(const nlohmann::json& j){
    std::string length_scale_string = lowercase_singular((std::string)j["mesh"]["time-scale"]);
    double time_factor = 1.0;
    switch (chash(length_scale_string)) {
        case chash<"planck-time">():
        case chash<"plancktime">():
        case chash<"pt">():
        case chash<"natural">():
            time_factor = funits::time<double, funits::planck_base>(1.0).convert_to<funits::SI_base>().count();
        break;
        case chash<"picosecond">():
            time_factor = 1e-12;
        break;
        case chash<"nanosecond">():
            time_factor = 1e-9;
        break;
        case chash<"microsecond">():
            time_factor = 1e-6;
        break;
        case chash<"millisecond">():
            time_factor = 1e-3;
        break;
        case chash<"second">():
            time_factor = 1.0;
        break;
        default:
            std::cerr << "Unrecognized time scale: " << (std::string)j["mesh"]["time-scale"] << "\n";
        break;
    }
    return time_factor;
}
double get_length_multiplier(const nlohmann::json& options){
    std::string length_scale_string = lowercase_singular((std::string)options["mesh"]["length-scale"]);
    double length_factor = 1.0;
    switch (chash(length_scale_string)) {
        case chash<"planck-length">():
        case chash<"plancklength">():
        case chash<"pl">():
        case chash<"natural">():
            length_factor = funits::length<double, funits::planck_base>(1.0).convert_to<funits::SI_base>().count();
        break;
        case chash<"picometer">():
            length_factor = 1e-12;
        break;
        case chash<"nanometer">():
            length_factor = 1e-9;
        break;
        case chash<"micrometer">():
            length_factor = 1e-6;
        break;
        case chash<"millimeter">():
            length_factor = 1e-3;
        break;
        case chash<"meter">():
            length_factor = 1.0;
        break;
        default:
            std::cerr << "Unrecognized length scale: " << (std::string)options["mesh"]["length-scale"] << "\n";
        break;
    }
    return length_factor;
}
config read_config(const char* filepath){
    std::ifstream cfile(filepath);
    nlohmann::json j;
    cfile >> j;
    config::scalar lmult = get_length_multiplier(j);
    config::scalar tmult = get_time_multiplier(j);
    config ret;

    ret.extents[0] = ((config::scalar)j["mesh"]["extents"][0] * lmult) / unit_length_in_meters;
    ret.extents[1] = ((config::scalar)j["mesh"]["extents"][1] * lmult) / unit_length_in_meters;
    ret.extents[2] = ((config::scalar)j["mesh"]["extents"][2] * lmult) / unit_length_in_meters;
    ret.resolution = getVector<uint32_t, 3>(j["mesh"]["resolution"]);

    //std::cerr << (std::string)j["mesh"]["time-scale"] << " " << tmult << " Tumult\n";
    //std::cerr << "Tmult: " << tmult << "\n";
    if(j.contains("timestep-ratio")){
        ret.timestep_ratio = (config::scalar)j["timestep-ratio"];
    }

    else{
        ret.timestep_ratio = 1;
    }
    ret.total_time = ((config::scalar)j["mesh"]["total-time"] * tmult) / unit_time_in_seconds;
    ret.space_charge = (bool)(j["mesh"]["space-charge"]);
    ret.bunch_gamma = (config::scalar)(j["bunch"]["gamma"]);
    if(ret.bunch_gamma < config::scalar(1)){
        std::cerr << "Gamma must be >= 1\n";
        exit(1);
    }
    assert(j.contains("undulator"));
    assert(j["undulator"].contains("static-undulator"));

    ret.undulator_K = j["undulator"]["static-undulator"]["undulator-parameter"];
    ret.undulator_period = ((config::scalar)j["undulator"]["static-undulator"]["period"] * lmult) / unit_length_in_meters;
    ret.undulator_length = ((config::scalar)j["undulator"]["static-undulator"]["length"] * lmult) / unit_length_in_meters;
    assert(!std::isnan(ret.undulator_length));
    assert(!std::isnan(ret.undulator_period));
    assert(!std::isnan(ret.extents[0]));
    assert(!std::isnan(ret.extents[1]));
    assert(!std::isnan(ret.extents[2]));
    assert(!std::isnan(ret.total_time));
    ret.length_scale_in_jobfile = get_length_multiplier(j);
    ret.temporal_scale_in_jobfile = get_time_multiplier(j);
    ret.charge = (config::scalar)j["bunch"]["charge"] * electron_charge_in_unit_charges;
    ret.mass = (config::scalar)j["bunch"]["mass"] * electron_mass_in_unit_masses;
    ret.num_particles = (uint64_t)j["bunch"]["number-of-particles"];
    ret.mean_position  = getVector<config::scalar, 3>(j["bunch"]["position"])                       * lmult / unit_length_in_meters;
    ret.sigma_position = getVector<config::scalar, 3>(j["bunch"]["sigma-position"])                 * lmult / unit_length_in_meters;
    ret.position_truncations = getVector<config::scalar, 3>(j["bunch"]["distribution-truncations"]) * lmult / unit_length_in_meters;
    ret.sigma_momentum = getVector<config::scalar, 3>(j["bunch"]["sigma-momentum"]);
    ret.output_rhythm = j["output"].contains("rhythm") ? uint32_t(j["output"]["rhythm"]) : 0;
    if(j.contains("experimentation")){
        nlohmann::json je = j["experimentation"];
        for(auto it = je.begin(); it!= je.end();it++){
            ret.experiment_options[it.key()] = double(it.value());
        }
    }
    return ret;
}