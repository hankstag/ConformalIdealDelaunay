#include "ConformalInterface.hh"
#include <igl/readOBJ.h>
#include <igl/readCSV.h>
#include <igl/matrix_to_list.h>
#include "argh.h"

int main(int argc, char* argv[]){

    auto cmdl = argh::parser(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);

    auto alg_params   = std::make_shared<AlgorithmParameters>();
    auto ls_params    = std::make_shared<LineSearchParameters>();
    auto stats_params = std::make_shared<StatsParameters>();
    
    // --d ${input_dir} --t ${input_dir}"/sphere10K_"$k"_Th_hat" --o ${output_dir} --p $k &
    std::string model, name, input_dir, out_dir, th_file;
    int max_itr = 10;
    bool use_mpf = false;
    cmdl("-in") >> model; // this parameter is for batch run
    cmdl("-d") >> input_dir;
    cmdl("-o") >> out_dir;
    cmdl("-m") >> max_itr;
    cmdl("-mpf") >> use_mpf;
    cmdl("-t") >> th_file;

    Eigen::MatrixXd Th_hat_mat;
    igl::readCSV(input_dir + "/" + th_file, Th_hat_mat);
    std::vector<double> Th_hat;
    igl::matrix_to_list(Th_hat_mat, Th_hat);

    int log_level = 1;
    cmdl("-l") >> log_level;

    name = model.substr(0, model.find_last_of('.'));
    name = name.substr(name.find_last_of('/')+1);
    
    stats_params->name = name;
    stats_params->error_log = false;
    stats_params->log_level = log_level;
    stats_params->print_summary = false;
    stats_params->output_dir = out_dir;
    
    alg_params->initial_ptolemy = true;
    alg_params->max_itr = max_itr;
    
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(input_dir + "/" + model, V, F);

    if(use_mpf){
        int mpf_prec = 100;
        alg_params->MPFR_PREC = mpf_prec;
        alg_params->min_lambda = std::pow(2, -100);
        alg_params->error_eps = 1e-24;
        alg_params->newton_decr_thres = -0.01 * alg_params->error_eps * alg_params->error_eps;
        spdlog::info("use mpf prec: {}", alg_params->MPFR_PREC);
    }else
        alg_params->error_eps = 1e-12;

    ls_params->bound_norm_thres = 1.0;

    std::vector<double> _pTh_hat(Th_hat.size(), 0.0);
    if(use_mpf){
      mpfr::mpreal::set_default_prec(alg_params->MPFR_PREC);
      mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
      mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
      // round Th_hat
      for(int i = 0; i < Th_hat.size(); i++){
        auto angle = Th_hat[i];
        _pTh_hat[i] = double(round(60 * angle / M_PI) * mpfr::const_pi() / 60);
      }
    }else{
      for(int i = 0; i < Th_hat.size(); i++)
        _pTh_hat[i] = double(Th_hat[i]);
    }

    std::vector<std::vector<double>> _pVn;
    std::vector<std::vector<int>> _pFn, _pFuv;
    std::vector<double> u, v;
    std::vector<int> p_Fn_to_F;

    std::vector<std::pair<int,int>> endpoints;
    std::tie(_pVn, _pFn, u, v, _pFuv, p_Fn_to_F, endpoints) = conformal_parametrization_VL<double>(V, F, _pTh_hat, alg_params, ls_params, stats_params);

    return 0;

}