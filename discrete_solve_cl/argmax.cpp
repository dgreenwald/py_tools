#include <eigen3/Eigen/Dense>
#include <iostream>
#include <math.h>
#define VIENNACL_WITH_EIGEN 1
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/maxmin.hpp"

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
  return m % x_i.size();
}

int main(){
  std::srand((unsigned int) time(0));
  Eigen::VectorXd x = Eigen::VectorXd::Random(10)*10;
  std::cout << x << '\n' << std::endl;
  std::cout << argmax(x, 3) << std::endl;
  
}
