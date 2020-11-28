#ifndef ETL_h
#define ETL_h

#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>

class ETL
{
    std::string dataset;
    std::string delimiter;
    bool header;

public:

    ETL(std::string data, std::string separator, bool head) : dataset(data), delimiter(separator), header(head)
    {}

    std::vector<std::vector<std::string>> readCSV();
    Eigen::MatrixXd CSVtoEigen(std::vector<std::vector<std::string>> dataset, int rows, int cols);

    Eigen::MatrixXd Normalize(Eigen::MatrixXd data, bool normalizeTarget);
    auto Mean(Eigen::MatrixXd data) -> decltype(data.colwise().mean());
    auto Std(Eigen::MatrixXd data) -> decltype(((data.array().square().colwise().sum())/(data.rows()-1)).sqrt());

    std::tuple<Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd,Eigen::MatrixXd> TrainTestSplit(Eigen::MatrixXd data, float train_size);

    void Vectortofile(std::vector<float> vector, std::string filename);
    void EigentoFile(Eigen::MatrixXd data, std::string filename);
};

#endif