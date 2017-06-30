#include <helper.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <eigen3/unsupported/Eigen/KroneckerProduct>
#define VIENNACL_WITH_EIGEN 1
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/linalg/maxmin.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/linalg/prod.hpp"


viennacl::matrix<double> eigen2vcl(Eigen::MatrixXd a){
  viennacl::matrix<double> x;
  viennacl::copy(a , x);
  return x;
}

viennacl::matrix<int> eigen2vcl(Eigen::MatrixXi a){
  viennacl::matrix<int> x;
  viennacl::copy(a , x);
  return x;
}

viennacl::matrix<float> eigen2vcl(Eigen::MatrixXf a){
  viennacl::matrix<float> x;
  viennacl::copy(a , x);
  return x;
}

viennacl::vector<double> eigen2vcl(Eigen::VectorXd a){
  viennacl::vector<double> x;
  viennacl::copy(a , x);
  return x;
}

viennacl::vector<int> eigen2vcl(Eigen::VectorXi a){
  viennacl::vector<int> x;
  viennacl::copy(a , x);
  return x;
}

viennacl::vector<float> eigen2vcl(Eigen::VectorXf a){
  viennacl::vector<float> x;
  viennacl::copy(a , x);
  return x;
}

int argmax(Eigen::VectorXd x, int approx){
  //Eigen matrices and vectors can be used directly in the solver
  Eigen::VectorXd x_a = x*pow(10, approx);
  Eigen::VectorXi x_i = x_a.cast<int>();
  x_i = x_i*x_i.size() + Eigen::VectorXi::LinSpaced(x_i.size(), 0, x_i.size()-1);
  viennacl::vector<int> x_v(x_i.size());
  viennacl::copy(x_i, x_v);
  viennacl::scalar<int> m_v;
  m_v = viennacl::linalg::max(x_v);
  int m;
  m = m_v;
  return (m % x_i.size() + x_i.size()) % x_i.size();
}

int discrete_solve_dense(double bet, const Eigen::MatrixXd flow, const Eigen::MatrixXd Pz, Eigen::MatrixXd* V, Eigen::VectorXi* indices){
  
  //initial setup of cpu objects
  int Nx = (*V).rows();
  int Nz = (*V).cols();
  int Ns = Nx*Nz;
  Eigen::MatrixXd bet_Pz = bet*Pz;
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Ns, Ns);
  Eigen::VectorXd f_star = Eigen::VectorXd::Zero(Ns);
  *indices = (-1 * Eigen::VectorXi::Ones(Ns));
  bool done = false;
  
  //copy needed objects to gpu
  //viennacl::matrix<double> bet_Pz_vcl(bet_Pz.rows(), bet_Pz.cols());
  //viennacl::copy(bet_Pz, bet_Pz_vcl);
  viennacl::matrix<double> Pz_vcl(Pz.rows(), Pz.cols());
  viennacl::copy(Pz, Pz_vcl);
  viennacl::matrix<double> V_vcl((*V).rows(), (*V).cols());
  viennacl::matrix<double> flow_vcl(flow.rows(), flow.cols());
  viennacl::copy(flow, flow_vcl);

  while(done == false){
    viennacl::copy((*V), V_vcl);
    Eigen::VectorXi old_indices = *indices;
    viennacl::matrix<double> Om = bet * viennacl::linalg::prod(V_vcl, trans(Pz_vcl));
    for(int jj = 0; jj < Nz; ++jj){
      Eigen::MatrixXd E(Nx, Nx);
      E = Eigen::MatrixXd::Zero(Nx, Nx);
	for(int ii=0; ii < Nx; ++ii){
	  int kk = Nx * jj + ii;
	  viennacl::vector<double> alternatives_vcl;
	  viennacl::vector<double> r = viennacl::row(flow_vcl, kk);
	  viennacl::vector<double> c = viennacl::column(Om, jj);
	  alternatives_vcl = r + c;
          Eigen::VectorXd alternatives(flow.cols());
          viennacl::copy(alternatives_vcl, alternatives);
          int max_ix = argmax(alternatives, 5);
	  double max_val = flow(kk, max_ix);
	  (*indices)(kk) = max_ix;
	  f_star(kk) = max_val;
	  E(ii, max_ix) = 1;
	}
        Eigen::MatrixXd l = bet_Pz.row(jj);
        A.block(Nx*jj, 0, Nx, A.cols()) = Eigen::kroneckerProduct(l, E);
    }
    Eigen::MatrixXd t = Eigen::MatrixXd::Identity(Ns,Ns) - A;
    Eigen::MatrixXd v = viennacl::linalg::solve(t, f_star, viennacl::linalg::gmres_tag(.001));
    v.resize(Nx, Nz);
    *V = v;
    done = (*indices).isApprox(old_indices);
  }
  return 0;
}

int discrete_solve_sparse(double bet,  Eigen::SparseMatrix<double> flow, const Eigen::SparseMatrix<double> Pz, Eigen::SparseMatrix<double>* V, Eigen::VectorXi* indices){

//initial setup of cpu objects
int Nx = (*V).rows();
int Nz = (*V).cols();
int Ns = Nx*Nz;
Eigen::SparseMatrix<double> bet_Pz = bet*Pz;
Eigen::MatrixXd A = Eigen::MatrixXd::Zero(Ns, Ns);
Eigen::VectorXd f_star = Eigen::VectorXd::Zero(Ns);
*indices = (-1 * Eigen::VectorXi::Ones(Ns));
bool done = false;

//copy needed objects to gpu
viennacl::compressed_matrix<double> bet_Pz_vcl_trans(bet_Pz.transpose().rows(), bet_Pz.transpose().cols());
Eigen::SparseMatrix<double> tran = bet_Pz.transpose();
viennacl::copy(tran, bet_Pz_vcl_trans);
viennacl::compressed_matrix<double> Pz_vcl(Pz.rows(), Pz.cols());
viennacl::copy(Pz, Pz_vcl);
viennacl::compressed_matrix<double> V_vcl((*V).rows(), (*V).cols());
viennacl::copy(*V, V_vcl);
viennacl::compressed_matrix<double> flow_vcl(flow.rows(), flow.cols());
viennacl::copy(flow, flow_vcl);
//viennacl::compressed_matrix<double> Pz_vcl_trans(Pz.transpose().rows(), Pz.transpose().cols());
//Eigen::SparseMatrix<double> tran = Pz.transpose();
//viennacl::copy(tran, Pz_vcl_trans);

while(done == false){
  viennacl::copy((*V), V_vcl);
  Eigen::VectorXi old_indices = *indices;
  Eigen::SparseMatrix<double> Om_E((*V).rows(), bet_Pz.transpose().cols());
  viennacl::compressed_matrix<double> Om = viennacl::linalg::prod(V_vcl, bet_Pz_vcl_trans);
  viennacl::copy(Om, Om_E);
  for(int jj = 0; jj < Nz; ++jj){
    Eigen::MatrixXd E(Nx, Nx);
    E = Eigen::MatrixXd::Zero(Nx, Nx);
    for(int ii=0; ii < Nx; ++ii){
      int kk = Nx * jj + ii;
      viennacl::vector<double> alternatives_vcl(flow.cols());
      viennacl::vector<double> r(flow.cols());
      viennacl::vector<double> c(flow.cols());
      Eigen::VectorXd r_E = flow.row(kk);
      Eigen::VectorXd c_E = Om_E.col(jj);
      viennacl::copy(r_E, r);
      viennacl::copy(c_E, c);
      alternatives_vcl = r + c;
      Eigen::VectorXd alternatives(flow.cols());
      viennacl::copy(alternatives_vcl, alternatives);
      int max_ix = argmax(alternatives, 4);
      double max_val = flow.coeffRef(kk, max_ix);
      (*indices)(kk) = max_ix;
      f_star(kk) = max_val;
      E(ii, max_ix) = 1;
    }
    Eigen::MatrixXd l = bet_Pz.row(jj);
    A.block(Nx*jj, 0, Nx, A.cols()) = Eigen::kroneckerProduct(l, E);
  }
  Eigen::MatrixXd t = Eigen::MatrixXd::Identity(Ns,Ns) - A;
  Eigen::MatrixXd v = viennacl::linalg::solve(t, f_star, viennacl::linalg::gmres_tag(.001));
  v.resize(Nx, Nz);
  *V = v.sparseView();
  done = (*indices).isApprox(old_indices);
}
return 0;
}

int main(){
Eigen::MatrixXd Pz;
Eigen::MatrixXd flow;
Eigen::MatrixXd V;
Eigen::VectorXi indices;
double bet = 0.98;
loadEigen("save/flow.out", flow);
loadEigen("save/Pz.out", Pz);
loadEigen("save/V.out", V);
Eigen::SparseMatrix<double> V_sparse = V.sparseView();
discrete_solve_dense(bet, flow, Pz, &V, &indices);
std::cout << V << "\n" << std::endl;
std::cout << indices << '\n' <<  std::endl;
Eigen::SparseMatrix<double> flow_sparse = flow.sparseView();
Eigen::SparseMatrix<double> Pz_sparse = Pz.sparseView();
discrete_solve_sparse(bet, flow_sparse, Pz_sparse, &V_sparse, &indices);
std::cout << V_sparse << "\n" << std::endl;
std::cout << indices << '\n' <<  std::endl;
}



