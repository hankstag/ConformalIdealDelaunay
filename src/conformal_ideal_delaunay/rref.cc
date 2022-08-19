#include "rref.hh"
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <Eigen/Core>
#include <igl/list_to_matrix.h>
#include <igl/Timer.h>

Eigen::VectorXi get_seq(int start, int end)
{
    Eigen::VectorXi seq(end - start + 1);
    for (int i = start; i <= end; i++)
    {
        seq(i - start) = i;
    }
    return seq;
}

template <typename mat>
void slice_mat(
    mat &A_in,
    int i_start, int i_end,
    int j_start, int j_end,
    mat &A_out)
{
    Eigen::VectorXi II = get_seq(i_start, i_end);
    Eigen::VectorXi JJ = get_seq(j_start, j_end);
    igl::slice(A_in, II, JJ, A_out);
}

void slice_into_sparse(
    Eigen::SparseMatrix<double> &A,
    const Eigen::VectorXi &II,
    const Eigen::VectorXi &JJ,
    Eigen::SparseMatrix<double> &B)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> dyn_A(A);
    // Iterate over rows
    for (int i = 0; i < B.rows(); i++)
    {
        // Iterate over cols
        for (int j = 0; j < B.cols(); j++)
        { 
            dyn_A.coeffRef(II(i),JJ(j)) = B.coeff(i, j);
        }
    }
    A = Eigen::SparseMatrix<double>(dyn_A);
}

void slice_into_mat(
    Eigen::SparseMatrix<double> &A,
    int i_start, int i_end,
    int j_start, int j_end,
    Eigen::SparseMatrix<double> &B)
{
    Eigen::VectorXi II = get_seq(i_start, i_end);
    Eigen::VectorXi JJ = get_seq(j_start, j_end);

    slice_into_sparse(A, II, JJ, B);
}

void slice_into_mat(
    Eigen::MatrixXd &A,
    int i_start, int i_end,
    int j_start, int j_end,
    Eigen::MatrixXd &B)
{
    Eigen::VectorXi II = get_seq(i_start, i_end);
    Eigen::VectorXi JJ = get_seq(j_start, j_end);
    igl::slice_into(B, II, JJ, A);
}


template <typename mat>
int find_pivot(
    mat &A,
    int i,
    int j,
    double &p)
{
    int m = A.rows() - 1;

    int k = i;
    p = A.coeff(i, j) * A.coeff(i, j);
    for (int l = i + 1; l <= m; l++)
    {
        double tmp = A.coeff(l, j) * A.coeff(l, j);
        if (tmp > p)
        {
            p = tmp;
            k = l;
        }
    }

    return k;
}

template <typename mat>
void rref(
    const mat &A_in,
    mat &R,
    std::vector<int> &jb,
    double tol)
{
    jb.clear();
    R = A_in;
    int m = R.rows() - 1, n = R.cols() - 1;

igl::Timer timer;
timer.start();
    // loop over the entire matrix
    int i = 0, j = 0;
    // Eigen::SparseMatrix<double, Eigen::RowMajor> R_rm(R);
    Eigen::SparseMatrix<double, Eigen::RowMajor> R_rm(R);
    while (i <= m && j <= n)
    {
        double p = 0;
        // find value(p) and index(k) of largest element in the remainder of column j
auto tm_pv0 = timer.getElapsedTime();        
        int k = find_pivot(R_rm, i, j, p);
        // int k = find_pivot(R_rm, i, j, p);
auto tm_pv1 = timer.getElapsedTime();

        std::cout << "k = " << k+1 << " p = " << p << std::endl;
std::cout << "find_pivot time: " << tm_pv1 - tm_pv0;
        if (p <= tol)
        {
            // the column is negligible, zero it out
            // R.prune([i, j](int ii, int jj, double) { return !(ii >= i && jj == j); });
            j++;
        }
        else
        {
            // remember column index
            jb.push_back(j);

auto tm_swap0 = timer.getElapsedTime();        
            // swap i-th and k-th rows
            // swap_two_rows(R, i, k);
            Eigen::SparseMatrix<double, Eigen::RowMajor> tmp_i = R_rm.row(i);
            Eigen::SparseMatrix<double, Eigen::RowMajor> tmp_k = R_rm.row(k);
    
            R_rm.row(i) = tmp_k;
            R_rm.row(k) = tmp_i;
auto tm_swap1 = timer.getElapsedTime();        
std::cout << "\tswap time: " << tm_swap1 - tm_swap0;
auto tm_subtract0 = timer.getElapsedTime();        
            // divide the pivot row by the pivot element
            Eigen::SparseMatrix<double, Eigen::RowMajor> Ai = R_rm.row(i) / R_rm.coeff(i, j);
            Eigen::SparseMatrix<double, Eigen::RowMajor> colj = R_rm.col(j);
auto tm_subtract1 = timer.getElapsedTime();        

            Eigen::SparseMatrix<double, Eigen::RowMajor> tmp = colj * Ai;

auto tm_subtract2 = timer.getElapsedTime();        

            R_rm = R_rm - tmp;
            R_rm.row(i) = Ai;

auto tm_subtract3 = timer.getElapsedTime();        
std::cout << "\tsubtract time: " << tm_subtract1 - tm_subtract0 << ", " << tm_subtract2 - tm_subtract1 << ", " << tm_subtract3 - tm_subtract2;
            i++;
            j++;
        }
auto tm_end = timer.getElapsedTime();        
std::cout << "\ttotal time:" << tm_end - tm_pv0 << std::endl;    
    }
    R = mat(R_rm);
}

void elim_constr(
    const Eigen::SparseMatrix<double> &C,
    Eigen::SparseMatrix<double> &T_out)
{
    int nvars = C.cols();
    std::vector<int> dep_list, indep_list;
    Eigen::SparseMatrix<double> R;
    rref(C, R, dep_list);

    // matlab code: indep = setdiff(1:nvars, dep);
    int k = 0;
    for (int i = 0; i < nvars; i++)
    {
        if (k == dep_list.size() || i != dep_list[k])
        {
            indep_list.push_back(i);
        }
        else
        {
            k++;
        }
    }
    std::cout << nvars << " dep: " << dep_list.size() << " indep: " << indep_list.size() << std::endl;
    Eigen::VectorXi dep, indep;
    igl::list_to_matrix(dep_list, dep);
    igl::list_to_matrix(indep_list, indep);
    Eigen::VectorXi all_rows = get_seq(0, R.rows() - 1);
    // std::cout << "dep list:" << dep << std::endl;
    // std::cout << "indep list:" << indep << std::endl;

    // matlab code: T = [-R(:,indep); speye(size(indep,2),size(indep,2))];
    Eigen::SparseMatrix<double> R_ind;
    igl::slice(R, all_rows, indep, R_ind);
    Eigen::SparseMatrix<double> T(R_ind.rows() + indep_list.size(), indep_list.size());
    T.reserve(R_ind.nonZeros() + indep.size());
    for (Eigen::Index c = 0; c < T.cols(); ++c)
    {
        T.startVec(c);

        for (typename Eigen::SparseMatrix<double>::InnerIterator itR_ind(R_ind, c); itR_ind; ++itR_ind)
        {
            T.insertBack(itR_ind.row(), c) = -itR_ind.value();
        }
        T.insertBack(c + R_ind.rows(), c) = 1;
    }
    T.finalize();

    // std::cout << "T after adding spyeye:\n" << T << std::endl;

    // matlab code: T([dep indep],:) = T;
    Eigen::VectorXi dep_indep(dep.size() + indep.size());
    Eigen::VectorXi all_cols_T;
    all_cols_T = get_seq(0, T.cols()-1);
    dep_indep << dep, indep;
    Eigen::VectorXi dep_indep_rev(dep_indep.size());
    for (int i = 0; i < dep_indep.size(); i++)
    {
        dep_indep_rev[dep_indep[i]] = i;
    }
    std::cout << "start slicing:" << std::endl;
    T_out.resize(T.rows(), T.cols());
    igl::slice(T, dep_indep_rev, 1, T_out);
    // slice_into_sparse(T_out, dep_indep, all_cols_T, T);
    // std::cout << "T output:\n" << T_out <<  std::endl;
}


void swap_two_rows(
    Eigen::SparseMatrix<double> &R,
    int i,
    int k
)
{
    // Eigen::SparseMatrix<double> tmp_i = R.row(i);
    // Eigen::SparseMatrix<double> tmp_k = R.row(k);
    // slice_into_mat(R, i, i, 0, R.cols()-1, tmp_k);
    // slice_into_mat(R, k, k, 0, R.cols()-1, tmp_i);

    Eigen::SparseMatrix<double, Eigen::RowMajor> R_rm(R);
    Eigen::SparseMatrix<double, Eigen::RowMajor> tmp_i = R_rm.row(i);
    Eigen::SparseMatrix<double, Eigen::RowMajor> tmp_k = R_rm.row(k);
    
    R_rm.row(i) = tmp_k;
    R_rm.row(k) = tmp_i;

    R = Eigen::SparseMatrix<double>(R_rm);
}

void elim_constr(
    const Eigen::SparseMatrix<double> &C,
    const Eigen::VectorXd &d,
    Eigen::SparseMatrix<double> &T_out,
    Eigen::VectorXd &b
)
{
    int nvars = C.cols();
    std::vector<int> dep_list, indep_list;
    Eigen::VectorXd rb;
    Eigen::SparseMatrix<double> R;
    Eigen::SparseMatrix<double> C_app = C;

    C_app.conservativeResize(C_app.rows(), C_app.cols() + 1);
    C_app.col(C_app.cols() - 1) = d.sparseView();
    rref(C_app, R, dep_list);

    rb = R.col(R.cols() - 1);
    R.conservativeResize(R.rows(), R.cols() - 1);
    // matlab code: indep = setdiff(1:nvars, dep);
    int k = 0;
    for (int i = 0; i < nvars; i++)
    {
        if (k == dep_list.size() || i != dep_list[k])
        {
            indep_list.push_back(i);
        }
        else
        {
            k++;
        }
    }
    std::cout << nvars << " dep: " << dep_list.size() << " indep: " << indep_list.size() << std::endl;
    Eigen::VectorXi dep, indep;
    igl::list_to_matrix(dep_list, dep);
    igl::list_to_matrix(indep_list, indep);
    Eigen::VectorXi all_rows = get_seq(0, R.rows() - 1);

    // matlab code: b = [rb; zeros(size(indep,2),1)]; 
    rb.conservativeResize(nvars);
    for (int i = dep_list.size(); i < nvars; i++)
    {
        rb(i) = 0;
    }
    // matlab code: T = [-R(:,indep); speye(size(indep,2),size(indep,2))];
    Eigen::SparseMatrix<double> R_ind;
    igl::slice(R, all_rows, indep, R_ind);
    Eigen::SparseMatrix<double> T(R_ind.rows() + indep_list.size(), indep_list.size());
    T.reserve(R_ind.nonZeros() + indep.size());
    for (Eigen::Index c = 0; c < T.cols(); ++c)
    {
        T.startVec(c);

        for (typename Eigen::SparseMatrix<double>::InnerIterator itR_ind(R_ind, c); itR_ind; ++itR_ind)
        {
            T.insertBack(itR_ind.row(), c) = -itR_ind.value();
        }
        T.insertBack(c + R_ind.rows(), c) = 1;
    }
    T.finalize();

    // matlab code: T([dep indep],:) = T;
    Eigen::VectorXi dep_indep(dep.size() + indep.size());
    Eigen::VectorXi all_cols_T;
    all_cols_T = get_seq(0, T.cols()-1);
    dep_indep << dep, indep;
    Eigen::VectorXi dep_indep_rev(dep_indep.size());
    for (int i = 0; i < dep_indep.size(); i++)
    {
        dep_indep_rev[dep_indep[i]] = i;
    }
    std::cout << "start slicing:" << std::endl;
    T_out.resize(T.rows(), T.cols());
    igl::slice(T, dep_indep_rev, 1, T_out);
    igl::slice(rb, dep_indep_rev, 1, b);
}