#include <iostream>

#include <Eigen/Dense>
#include <Eigen/LU>

#include "JACOB.hpp"

using matrix16   = Eigen::Matrix<float, 16, 16>;

int main() {
    auto X0 = 0;
    auto X1 = 1;
    auto Y0 = 0;
    auto Y1 = 1;
    matrix16 JACOBIAN;
    INIT_JACOBIAN(X0, X1, Y0, Y1, JACOBIAN);

    std::cout << JACOBIAN << std::endl;
    return 0;
}
