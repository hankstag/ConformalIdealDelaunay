#include "ConformalInterface.hh"
#include <igl/readOBJ.h>
#include <igl/readCSV.h>
#include <igl/matrix_to_list.h>
#include "argh.h"
#include "plot.hh"
#include <igl/list_to_matrix.h>

#include <igl/euler_characteristic.h>
#include <igl/vertex_components.h>
#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/adjacency_matrix.h>
#include <igl/boundary_loop.h>
#include <igl/edges.h>

#include <igl/opengl/glfw/Viewer.h>

// return n_genus and n_bd
std::pair<int,int> topology_info_of_mesh(const Eigen::MatrixXi& F){
  Eigen::MatrixXi C, Ci;
  Eigen::SparseMatrix<double> AL;
  igl::adjacency_matrix(F, AL);
  igl::vertex_components(AL, C, Ci);
  int connected_components = Ci.rows();
  // #V should be the total number of all non-detached vertices
  int nv = 0;
  for(int i=0;i<Ci.rows();i++){
    if(Ci(i) != 1)
      nv += Ci(i);
  }
  int nf = F.rows();
  Eigen::MatrixXi E;
  igl::edges(F, E);

  // int eu = igl::euler_characteristic(F);
  std::vector<std::vector<int>> bds;
  igl::boundary_loop(F, bds);
  // check whether an edge is shared by more than two facets
  bool is_edge_manifold = igl::is_edge_manifold(F);
  Eigen::VectorXi VB;
  igl::is_vertex_manifold(F, VB);

  int n_bd = bds.size();
  int ne = E.rows();
  int eu = nv - ne + nf + n_bd;
  int n_genus = (2-eu)/2;
  return std::make_pair(n_genus, n_bd);
}


int main(int argc, char* argv[]){

    auto cmdl = argh::parser(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);

    auto alg_params   = std::make_shared<AlgorithmParameters>();
    auto ls_params    = std::make_shared<LineSearchParameters>();
    auto stats_params = std::make_shared<StatsParameters>();
    
    // --d ${input_dir} --t ${input_dir}"/sphere10K_"$k"_Th_hat" --o ${output_dir} --p $k &
    std::string model, name, input_dir, out_dir;
    int max_itr = 10;
    bool use_mpf = false;
    cmdl("-in") >> model; // this parameter is for batch run
    cmdl("-d") >> input_dir;
    cmdl("-o") >> out_dir;
    cmdl("-m") >> max_itr;
    cmdl("-mpf") >> use_mpf;
    cmdl("-x") >> alg_params->use_xi;

    name = model.substr(0, model.find_last_of('.'));
    name = name.substr(name.find_last_of('/')+1);

    Eigen::MatrixXd Th_hat_mat;
    igl::readCSV(input_dir + "/" + name + "_Th_hat", Th_hat_mat);
    std::vector<double> Th_hat;
    igl::matrix_to_list(Th_hat_mat, Th_hat);

    int log_level = 1;
    cmdl("-l") >> log_level;
    
    stats_params->name = name;
    stats_params->error_log = false;
    stats_params->log_level = log_level;
    stats_params->print_summary = true;
    stats_params->output_dir = out_dir;
    
    alg_params->initial_ptolemy = true;
    alg_params->max_itr = max_itr;
    alg_params->bypass_overlay = false;
    
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::readOBJ(input_dir + "/" + model, V, F);

    if(use_mpf){
        int mpf_prec = 300;
        alg_params->MPFR_PREC = mpf_prec;
        alg_params->min_lambda = std::pow(2, -100);
        alg_params->error_eps = 1e-24;
        alg_params->newton_decr_thres = -0.01 * alg_params->error_eps * alg_params->error_eps;
        spdlog::info("use mpf prec: {}", alg_params->MPFR_PREC);
    }else
        alg_params->error_eps = 1e-12;

    ls_params->bound_norm_thres = 1.0;

    std::vector<double> _pTh_hat(Th_hat.size(), 0.0);
    std::vector<mpfr::mpreal> _pTh_hat_mpf(Th_hat.size(), 0.0);
    if(use_mpf){
      mpfr::mpreal::set_default_prec(alg_params->MPFR_PREC);
      mpfr::mpreal::set_emax(mpfr::mpreal::get_emax_max());
      mpfr::mpreal::set_emin(mpfr::mpreal::get_emin_min());
      // round Th_hat
      for(int i = 0; i < Th_hat.size(); i++){
        auto angle = Th_hat[i];
        _pTh_hat[i] = double(round(60 * angle / M_PI) * mpfr::const_pi() / 60);
        _pTh_hat_mpf[i] = mpfr::mpreal(round(60 * angle / M_PI) * mpfr::const_pi() / 60);
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
    
    // setup gamma loops for the seamless similarity condition
    std::vector<std::vector<int>> gamma = {{}};
    std::vector<double> kappa_hat;
    std::vector<mpfr::mpreal> kappa_hat_mpf;

    Eigen::MatrixXi gamma_mat;
    igl::readCSV(input_dir + "/" + name + "_gamma", gamma_mat);
    for(int i = 0; i < gamma_mat.rows(); i++){
        if(gamma_mat(i, 0) == -1)
            gamma.push_back({});
        else
            gamma.back().push_back(gamma_mat(i));
    }
    Eigen::MatrixXd kappa_hat_mat;
    igl::readCSV(input_dir + "/" + name + "_kappa_hat", kappa_hat_mat);
    for(int i = 0; i < kappa_hat_mat.rows(); i++)
        kappa_hat.push_back(kappa_hat_mat(i, 0));

// #define VIEW_GAMMA_LOOPS
#ifdef VIEW_GAMMA_LOOPS
    igl::opengl::glfw::Viewer viewer;
    viewer.data().set_mesh(V, F);
    Eigen::MatrixXd colors(F.rows(), 3); colors.setConstant(1);
    for(int f: gamma[1])
        colors.row(f) << 1, 0, 0;
    // colors.row(gamma[0][0]) << 0, 0, 1;
    // colors.row(gamma[0][1]) << 0, 1, 0;
    viewer.data().set_colors(colors);
    for(int i = 0; i < Th_hat.size(); i++){
        if(Th_hat[i] != M_PI * 2)
            viewer.data().add_points(V.row(i), Eigen::RowVector3d(1, 0, 0));
    }
    viewer.launch();
#endif

    auto gb = topology_info_of_mesh(F);
    if(std::get<0>(gb) == 0){
        gamma.clear();
        kappa_hat.clear();
    }

    if(use_mpf){
        std::cout << "use mpf!!\n";
        for(int i = 0; i < kappa_hat.size(); i++)
            kappa_hat_mpf.push_back(mpfr::mpreal(kappa_hat[i]));
        auto res = conformal_parametrization_VL<mpfr::mpreal>(V, F, _pTh_hat_mpf, gamma, kappa_hat_mpf, alg_params, ls_params, stats_params);
    }else
        std::tie(_pVn, _pFn, u, v, _pFuv, p_Fn_to_F, endpoints) = conformal_parametrization_VL<double>(V, F, _pTh_hat, gamma, kappa_hat, alg_params, ls_params, stats_params);

    Eigen::MatrixXi pFn, pFuv;
    Eigen::MatrixXd pVn;
    igl::list_to_matrix(_pVn, pVn);
    igl::list_to_matrix(_pFn, pFn);
    igl::list_to_matrix(_pFuv, pFuv);
    Eigen::MatrixXd p_uv(u.size(), 2);
    for(int i = 0; i < u.size(); i++)
      p_uv.row(i) << u[i], v[i];

    double scale = 1.0;
    bool show_uv = false;
    igl::opengl::glfw::Viewer viewer;
    viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &, unsigned int key, int mod)
    {
        if(key == '1'){
            scale *= 2;
            viewer.data().set_uv(scale * p_uv, pFuv);
        }else if(key == '2'){
            scale /= 2.0;
            viewer.data().set_uv(scale * p_uv, pFuv);
        }else if(key == '3'){
            viewer.data().clear();
            show_uv = !show_uv;
            if(show_uv){
                viewer.data().set_mesh(p_uv, pFuv);
                viewer.core().align_camera_center(p_uv);
            }else
                viewer.data().set_mesh(pVn, pFn);
        }
        return false;
    };

    viewer.data().set_mesh(pVn, pFn);
    viewer.data().set_uv(p_uv, pFuv);
    viewer.data().show_texture = true;
    viewer.launch();

    return 0;

}