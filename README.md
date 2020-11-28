# Machine Learning with C++ Tutorial

In this repository, you can find all the code from my series of tutorials of Machine Learning with C++: [YouTube Playlist](https://www.youtube.com/watch?v=jKtbNvCT8Dc&list=PLNpKaH98va-FJ1YN8oyMQWnR1pKzPu-GI).

# Usage

Fork and clone/download the repository. 

## Linear Regression

To compile simply run the code:

`g++ -std=c++11 LinearRegression/LinearRegression.cpp ETL/ETL.cpp main/LinearRegression.cpp -o linregr`

To run and test:

`./linregr datasets/winedata.csv ","`

## Logistic Regression

To compile simply run the code:

`g++ -std=c++11 LogisticRegression/LogisticRegression.cpp ETL/ETL.cpp main/LogisticRegression.cpp -o logregr`

To run and test:

`./logregr datasets/adult_data.csv ","`