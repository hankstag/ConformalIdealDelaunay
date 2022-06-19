#ifndef FORM_CONVERSION_H
#define FORM_CONVERSION_H

#include <Eigen/Core>
#include "OverlayMesh.hh"

namespace OverlayProblem{

  template <typename Scalar>
  void form_conversion(Mesh<Scalar> & m, const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& xi, Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& phi);

}

#endif