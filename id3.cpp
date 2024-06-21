#include <bits/stdc++.h>

using namespace std;


class Node {
public:
    int attribute;
    double value;
    int result;
    map<double, Node*> children;

    Node(int attr = -1, double val = 0.0, int res = -1) : attribute(attr), value(val), result(res) {}
};

vector<vector<double>> readCSV(const string& filename) {
    vector<vector<double>> data;
    ifstream file(filename);
    string line, cell;
    
    while (getline(file, line)) {
        stringstream lineStream(line);
        vector<double> row;
        while (getline(lineStream, cell, ',')) {
            row.push_back(stod(cell));
        }
        data.push_back(row);
    }
    
    return data;
}


vector<double> getColumn(const vector<vector<double>>& data, int col) {
    vector<double> column;
    for (const auto& row : data) {
        column.push_back(row[col]);
    }
    return column;
}

double entropy(const vector<int>& y) {
    map<int, int> counts;
    for (int value : y) {
        counts[value]++;
    }
    
    double entropy = 0.0;
    for (const auto& [key, count] : counts) {
        double probability = static_cast<double>(count) / y.size();
        entropy -= probability * log2(probability);
    }
    
    return entropy;
}

double informationGain(const vector<int>& y, const vector<vector<int>>& splits) {
    double total_entropy = entropy(y);
    double weighted_entropy = 0.0;
    
    for (const auto& split : splits) {
        double split_entropy = entropy(split);
        weighted_entropy += static_cast<double>(split.size()) / y.size() * split_entropy;
    }
    
    return total_entropy - weighted_entropy;
}


int main() {
    return 0;
}