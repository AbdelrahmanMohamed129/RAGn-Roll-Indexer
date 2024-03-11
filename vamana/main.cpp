//
// Created by farah on 3/4/2024.
//
#include <bits/stdc++.h>
#include "ragnroll_vamana.h"

using namespace std;

int main() {
    const int dim = 768;
    auto x = new Vamana(12,15,1.2,dim);
    vector<vector<float>>pdata;
    ifstream myFile("cluster0.txt");

    if (myFile.is_open()) {
        while (myFile) {
            string line;
            getline (myFile, line);
            if(!line.size()) break;
            line.erase(remove(line.begin(), line.end(), '['), line.end());
            line.erase(remove(line.begin(), line.end(), ']'), line.end());
            pdata.push_back(vector<float>());
            stringstream stream_line(line);
            while (stream_line.good()) {
                string substr;
                getline(stream_line, substr, ' ');
                if(!substr.length()) continue;
                pdata.back().push_back(stof(substr));
            }
        }
    }
    x->CreateIndex(pdata);
    return 0;
}

