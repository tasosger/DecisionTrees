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