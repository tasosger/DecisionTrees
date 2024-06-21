#include <bits/stdc++.h>

using namespace std;


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
    
    // Skip the header line
    getline(file, line);
    
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

bool isNumeric(const string& str) {
    try {
        stod(str);
    } catch (const invalid_argument&) {
        return false;
    } catch (const out_of_range&) {
        return false;
    }
    return true;
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

Node* id3(const vector<vector<double>>& X, const vector<int>& y, const set<int>& attributes) {
    if (set<int>(y.begin(), y.end()).size() == 1) {
        return new Node(-1, 0.0, y[0]);
    }
    
    if (attributes.empty()) {
        int most_common = *max_element(y.begin(), y.end(), 
                            [&y](int a, int b) { return count(y.begin(), y.end(), a) < count(y.begin(), y.end(), b); });
        return new Node(-1, 0.0, most_common);
    }
    
    int best_attribute = -1;
    double max_gain = -1;
    
    for (int attribute : attributes) {
        map<double, vector<int>> splits_map;
        
        for (size_t i = 0; i < X.size(); ++i) {
            splits_map[X[i][attribute]].push_back(y[i]);
        }
        
        vector<vector<int>> splits;
        for (const auto& [key, split] : splits_map) {
            splits.push_back(split);
        }
        
        double gain = informationGain(y, splits);
        
        if (gain > max_gain) {
            max_gain = gain;
            best_attribute = attribute;
        }
    }
    
    if (max_gain == 0) {
        int most_common = *max_element(y.begin(), y.end(), 
                            [&y](int a, int b) { return count(y.begin(), y.end(), a) < count(y.begin(), y.end(), b); });
        return new Node(-1, 0.0, most_common);
    }
    
    Node* node = new Node(best_attribute);
    set<int> remaining_attributes = attributes;
    remaining_attributes.erase(best_attribute);
    
    map<double, vector<int>> splits_map;
    map<double, vector<vector<double>>> X_splits_map;
    
    for (size_t i = 0; i < X.size(); ++i) {
        splits_map[X[i][best_attribute]].push_back(y[i]);
        X_splits_map[X[i][best_attribute]].push_back(X[i]);
    }
    
    for (const auto& [value, split] : splits_map) {
        node->children[value] = id3(X_splits_map[value], split, remaining_attributes);
    }
    
    return node;
}

int predict(Node* node, const vector<double>& sample, int default_class) {
    if (node->result != -1) {
        return node->result;
    }
    
    auto it = node->children.find(sample[node->attribute]);
    if (it != node->children.end()) {
        return predict(it->second, sample, default_class);
    }
    
    return default_class;
}

pair<double, vector<vector<int>>> calculate_accuracy(Node* tree, const vector<vector<double>>& X_test, const vector<int>& y_test, int default_class) {
    vector<int> predictions;
    for (const auto& sample : X_test) {
        predictions.push_back(predict(tree, sample, default_class));
    }
    
    int correct = 0;
    for (size_t i = 0; i < y_test.size(); ++i) {
        if (predictions[i] == y_test[i]) {
            correct++;
        }
    }
    
    int num_classes = 2; // Since we only have two classes: 0 and 1
    vector<vector<int>> confusion_matrix(num_classes, vector<int>(num_classes, 0));
    
    for (size_t i = 0; i < y_test.size(); ++i) {
        confusion_matrix[y_test[i]][predictions[i]]++;
    }
    
    return { static_cast<double>(correct) / y_test.size(), confusion_matrix };
}

vector<vector<double>> encodeData(const vector<vector<string>>& rawData, vector<int>& y, unordered_map<string, int>& encodings) {
    int next_code = 0;
    vector<vector<double>> encodedData;
    
    for (const auto& row : rawData) {
        vector<double> encodedRow;
        for (size_t i = 0; i < row.size() - 1; ++i) {
            if (isNumeric(row[i])) {
                encodedRow.push_back(stod(row[i]));
            } else {
                if (encodings.find(row[i]) == encodings.end()) {
                    encodings[row[i]] = next_code++;
                }
                encodedRow.push_back(encodings[row[i]]);
            }
        }
        encodedData.push_back(encodedRow);
        
        // Handle the last column as the target variable
        if (row.back() == "Approved") {
            y.push_back(1); // Encode Approved as 1
        } else if (row.back() == "Denied") {
            y.push_back(0); // Encode Denied as 0
        } else {
            cerr << "Unexpected value in the target column: " << row.back() << endl;
            exit(EXIT_FAILURE);
        }
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
    unordered_map<string, int> encodings;
    vector<vector<double>> X = encodeData(rawData, y, encodings);

    // Output some of the data to verify it's read correctly
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

    // Output the size of the dataset
    cout << "Total number of samples: " << X.size() << endl;

    // Output encodings for the features
    cout << "Feature encodings:" << endl;
    for (const auto& [key, value] : encodings) {
        cout << key << ": " << value << endl;
    }

    set<int> attributes;
    for (size_t i = 0; i < X[0].size(); ++i) {
        attributes.insert(i);
    }

    // Split the data into training and test sets (70% training, 30% test)
    vector<vector<double>> X_train, X_test;
    vector<int> y_train, y_test;

    srand(static_cast<unsigned int>(time(0)));  // Seed the random number generator
    for (size_t i = 0; i < X.size(); ++i) {
        if (static_cast<double>(rand()) / RAND_MAX > 0.3) {
            X_train.push_back(X[i]);
            y_train.push_back(y[i]);
        } else {
            X_test.push_back(X[i]);
            y_test.push_back(y[i]);
        }
    }

    cout << "Training set size: " << X_train.size() << endl;
    cout << "Test set size: " << X_test.size() << endl;

    Node* root = id3(X_train, y_train, attributes);
    int default_class = *max_element(y_train.begin(), y_train.end(), 
                        [&y_train](int a, int b) { return count(y_train.begin(), y_train.end(), a) < count(y_train.begin(), y_train.end(), b); });

    auto [accuracy, conf_matrix] = calculate_accuracy(root, X_test, y_test, default_class);

    cout << "Accuracy: " << accuracy << endl;
    cout << "Confusion Matrix:" << endl;
    for (const auto& row : conf_matrix) {
        for (int val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}