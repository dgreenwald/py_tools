#include <eigen3/Eigen/Dense>
#include <iostream>
#define VIENNACL_WITH_EIGEN 1
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/gmres.hpp"

Eigen::VectorXd solve_dense(Eigen::MatrixXd A, Eigen::VectorXd b){
  //Eigen matrices and vectors can be used directly in the solver
  return viennacl::linalg::solve(A, b, viennacl::linalg::gmres_tag());
}

int main(){
  std::srand((unsigned int) time(0));
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(3,3)*100;
  Eigen::VectorXd b = Eigen::VectorXd::Random(3)*100;
  std::cout << A << '\n' << b << '\n' << std::endl;
  std::cout << solve_dense(A, b) << '\n' << std::endl;
  std::cout << A*solve_dense(A, b) << std::endl;
}

   
