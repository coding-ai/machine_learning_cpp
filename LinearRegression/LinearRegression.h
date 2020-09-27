#ifndef LinearRegression_h
#define LinearRegression_h

#include <eigen3/Eigen/Dense>

class LinearRegression
{

public:
    LinearRegression()
    {}

    float OLS_Cost(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::MatrixXd theta);
    std::tuple<Eigen::VectorXd,std::vector<float>> GradientDescent(Eigen::MatrixXd X, Eigen::MatrixXd y, Eigen::VectorXd theta, float alpha, int iters);
    float RSquared(Eigen::MatrixXd y, Eigen::MatrixXd y_hat);
};

#endif