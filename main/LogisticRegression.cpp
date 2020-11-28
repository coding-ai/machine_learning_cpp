#include "../ETL/ETL.h"
#include "../LogisticRegression/LogisticRegression.h"

#include <vector>
#include <string>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <list>

int main(int argc, char *argv[]) {

    ETL etl(argv[1], argv[2], argv[3]);

    std::vector<std::vector<std::string>> dataset = etl.readCSV();

    int rows = dataset.size();
    int cols = dataset[0].size();

    Eigen::MatrixXd dataMat = etl.CSVtoEigen(dataset, rows, cols);

    Eigen::MatrixXd norm = etl.Normalize(dataMat, false);

    Eigen::MatrixXd X_train, y_train, X_test, y_test;
    std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> split_data = etl.TrainTestSplit(norm, 0.8);
    std::tie(X_train, y_train, X_test, y_test) = split_data;

    LogisticRegression lr;

    int dims = X_train.cols();
    Eigen::MatrixXd W = Eigen::VectorXd::Zero(dims);
    double b = 0.0;
    double lambda = 0.0;
    bool log_cost = true;
    double learning_rate = 0.01;
    int num_iter = 10000;

    Eigen::MatrixXd dw;
    double db;
    std::list<double> costs;
    std::tuple<Eigen::MatrixXd, double, Eigen::MatrixXd, double, std::list<double>> optimize = lr.Optimize(W, b, X_train, y_train, num_iter, learning_rate, lambda, log_cost);
    std::tie(W,b,dw,db,costs) = optimize;

    Eigen::MatrixXd y_pred_test = lr.Predict(W,b,X_test);
    Eigen::MatrixXd y_pred_train = lr.Predict(W,b,X_train);

    auto train_acc = (100-(y_pred_train-y_train).cwiseAbs().mean()*100);
    auto test_acc = (100-(y_pred_test-y_test).cwiseAbs().mean()*100);

    std::cout << "Train Accuracy: " << train_acc << std::endl;
    std::cout << "Test Accuracy: " << test_acc << std::endl;

    //std::vector<float> costVec(costs.begin(), costs.end());
    //etl.Vectortofile(costVec,"datasets/cost.txt");

    return EXIT_SUCCESS;
}