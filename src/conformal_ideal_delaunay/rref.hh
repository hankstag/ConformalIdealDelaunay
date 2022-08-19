#ifndef RREF_H
#define RREF_H
#include <Eigen/Sparse>
#include <iostream>

void swap_two_rows(
    Eigen::SparseMatrix<double> &R,
    int i,
    int k
);
template<typename mat>
void rref(
    const mat &A_in,
    mat &R,
    std::vector<int> &jb,
    double tol = 2e-12);

void elim_constr(
    const Eigen::SparseMatrix<double> &C,
    Eigen::SparseMatrix<double> &T
);

void elim_constr(
    const Eigen::SparseMatrix<double> &C,
    const Eigen::VectorXd &d,
    Eigen::SparseMatrix<double> &T_out,
    Eigen::VectorXd &b
);
#endif