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

vector<vector<string>> readCSV(const string& filename) {
    vector<vector<string>> data;
    ifstream file(filename);
    string line, cell;
    
    while (getline(file, line)) {
        stringstream lineStream(line);
        vector<string> row;
        while (getline(lineStream, cell, ',')) {
            row.push_back(cell);
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
vector<vector<double>> encodeData(const vector<vector<string>>& rawData, vector<int>& y) {
    unordered_map<string, int> encodings;
    int next_code = 0;
    vector<vector<double>> encodedData;
    
    for (const auto& row : rawData) {
        vector<double> encodedRow;
        for (size_t i = 0; i < row.size() - 1; ++i) {
            if (encodings.find(row[i]) == encodings.end()) {
                encodings[row[i]] = next_code++;
            }
            encodedRow.push_back(encodings[row[i]]);
        }
        encodedData.push_back(encodedRow);
        
        // Handle the last column as the target variable
        if (encodings.find(row.back()) == encodings.end()) {
            encodings[row.back()] = next_code++;
        }
        y.push_back(encodings[row.back()]);
    }
    
    return encodedData;
}


int main() {
    string filename = "./loan.csv";
    vector<vector<string>> rawData = readCSV(filename);
    
    if (rawData.empty()) {
        cout << "Failed to read the CSV file or the file is empty." << endl;
        return -1;
    }

    vector<int> y;
    vector<vector<double>> X = encodeData(rawData, y);

   
    cout << "First 5 rows of X:" << endl;
    for (size_t i = 0; i < min(size_t(5), X.size()); ++i) {
        for (size_t j = 0; j < X[i].size(); ++j) {
            cout << X[i][j] << " ";
        }
        cout << endl;
    }

    cout << "First 5 elements of y:" << endl;
    for (size_t i = 0; i < min(size_t(5), y.size()); ++i) {
        cout << y[i] << " ";
    }
    cout << endl;

    set<int> attributes;
    for (size_t i = 0; i < X[0].size(); ++i) {
        attributes.insert(i);
    }

    return 0;
}