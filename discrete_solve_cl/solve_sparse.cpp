#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#define VIENNACL_WITH_EIGEN 1
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/gmres.hpp"
#include "viennacl/compressed_matrix.hpp"


Eigen::VectorXd solve_sparse(Eigen::SparseMatrix<double> A, Eigen::VectorXd b){
  //Eigen matrices and vectors can be used directly in the solver
  return viennacl::linalg::solve(A, b, viennacl::linalg::gmres_tag());
}

int main(){
  std::srand((unsigned int) time(0));
  Eigen::MatrixXd r = Eigen::MatrixXd::Random(3,3);
  Eigen::SparseMatrix<double> A = r.sparseView();
  Eigen::VectorXd b = Eigen::VectorXd::Random(3);
  std::cout << A << '\n' << b << '\n' << std::endl;
  std::cout << solve_sparse(A, b) << '\n' << std::endl;
  std::cout << A*solve_sparse(A, b) << '\n' << std::endl;
  std::srand((unsigned int) time(0)*2);
  Eigen::MatrixXd x = Eigen::MatrixXd::Random(3,3);
  Eigen::SparseMatrix<double> X = x.sparseView();
  Eigen::VectorXd v = Eigen::VectorXd::Random(3);
  std::cout << x << '\n' << std::endl;
  viennacl::compressed_matrix<double> x_1v;
  viennacl::vector<double> v_v(3);
  viennacl::copy(X, x_1v);
  viennacl::copy(v, v_v);
  std::cout << viennacl::linalg::prod(x_1v, v_v) << std::endl;
}
