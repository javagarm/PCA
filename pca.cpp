#include <iostream>
#include<Vector>
#include<Eigen/Dense>
#include<fstream>
using namespace std;
using namespace Eigen;
//To store the output of PCA into a csv file
void toCsv(string fileName, MatrixXd  matrix)
{
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
    ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(CSVFormat);
        file.close();
    }
}
//To extract the data from csv and save it in a matrix format
MatrixXd toEigen(string file)
{
    vector<double> entries;
    // in this object we store the data from the matrix
    ifstream dataFile(file);
    // this variable is used to store the row of the matrix that contains commas
    string rowString;
    // this variable is used to store the matrix entry;
    string entry;
    // this variable is used to track the number of rows
    int row = 0;
    while (getline(dataFile, rowString)) // here we read a row by row of matrixDataFile and store every line into the string variable matrixRowString
    {
        stringstream rowStringStream(rowString); //convert matrixRowString that is a string to a stream variable.

        while (getline(rowStringStream, entry, ',')) // here we read pieces of the stream matrixRowStringStream until every comma, and store the resulting character into the matrixEntry
        {
            entries.push_back(stod(entry));   //here we convert the string to double and fill in the row vector storing all the matrix entries
        }
        row++; //update the column numbers
    }
    // here we convert the vector variable into the matrix
    // matrixEntries.data() is the pointer to the first memory location at which the entries of the vector matrixEntries are stored;
    return Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(entries.data(), row, entries.size() / row);
}

MatrixXd pca(MatrixXd  matrix,int d){
    int i,j;
    //Matrix to save the output of pca
    VectorXd outputMatrix;
    MatrixXd normalized;
    int r = matrix.rows();
    int c = matrix.cols();

    cout<<"rowsXcols:\t"<<r<<"x"<<c;
    normalized = matrix;
    normalized.normalize();
    //PCA Algorithm
    MatrixXd mean;
    MatrixXd centered;
    MatrixXd cov;
    VectorXd eigen_values;
    MatrixXd eigen_vectors;
    centered = normalized.rowwise()-normalized.colwise().mean();
    mean = normalized - centered;
    cov = (centered.adjoint()*centered)/(normalized.rows()-1);
    SelfAdjointEigenSolver<MatrixXd> eigen_solver(cov);
	eigen_values = eigen_solver.eigenvalues();
    eigen_vectors = eigen_solver.eigenvectors();
    sort(eigen_values.derived().data(), eigen_values.derived().data() + eigen_values.derived().size());
    short index = eigen_values.size() - 1;
    MatrixXd featureVector = eigen_vectors.rightCols(d);
    return featureVector;
}
int main(){
    //Out put Matrix
    MatrixXd output;
    int d;
    cout<<"No of dimensions : ";
    cin>>d;
    // matrix to be loaded from a file
    MatrixXd input;
    // load the matrix from the file
    input = toEigen("titanic train.csv");
    output = pca(input,d);
    output.rowwise().reverse();
    output = input*output;
    //save the output to csv file
    toCsv("output_1.csv",output);
    return 0;
}
