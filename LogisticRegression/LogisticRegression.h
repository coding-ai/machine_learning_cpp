#ifndef LogisticRegression_h
#define LogisticRegression_h

#include <eigen3/Eigen/Dense>
#include <list>

class LogisticRegression
{
public:
    LogisticRegression()
    {}

    Eigen::MatrixXd Sigmoid(Eigen::MatrixXd Z);

    std::tuple<Eigen::MatrixXd, double, double> Propagate(Eigen::MatrixXd W, double b, Eigen::MatrixXd X, Eigen::MatrixXd y, double lambda);
    std::tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double, std::list<double>> Optimize(Eigen::MatrixXd W, double b, Eigen::MatrixXd X, Eigen::MatrixXd y, int num_iter, double learning_rate, double lambda, bool log_cost);
    Eigen::MatrixXd Predict(Eigen::MatrixXd W, double b, Eigen::MatrixXd X);
};

#endif