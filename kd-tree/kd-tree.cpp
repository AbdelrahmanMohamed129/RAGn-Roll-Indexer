//
// Created by farah on 2/16/2024.
//

#include <iostream>
#include <bits/stdc++.h>
#include <array>
#include <vector>
#include <fstream>

#include "kd-tree.h"

using namespace std;

// user-defined point type
// inherits std::array in order to use operator[]
class MyPoint : public std::array<double, 768>
{
public:

    // dimension of space (or "k" of k-d tree)
    // KDTree class accesses this member
    static const int DIM = 768;

    // the constructors
    MyPoint() {}

    MyPoint(vector<double> &dims)
    {
        for (int i = 0; i < DIM; ++i) {
            (*this)[i] = dims[i];
        }
    }
};

int cnt = 0;

void mapClusters(int batch) {
    string batchFileName = "batches/dataset/"+to_string(batch)+"-embeds-batch.txt";
    ifstream batchFile(batchFileName);

    vector<vector<double>> embeds;
    if (batchFile.is_open()) {
        while (batchFile) {
            string line;
            getline (batchFile, line);
            if(!line.size()) break;
            embeds.push_back(vector<double>());
            stringstream stream_line(line);
            while (stream_line.good()) {
                string substr;
                getline(stream_line, substr, ' ');
                if(substr.length() < 2) continue;
                embeds.back().push_back(stod(substr));
            }
        }
    }

    map<int,vector<pair<int, vector<double>>>> labels;
    string labelFileName = "batches/labels/labels"+to_string(batch)+".txt";
    ifstream labelFile(labelFileName);
    ofstream outfile;

    if (labelFile.is_open()) {
        while (labelFile) {
            int i(0);
            string line;
            getline (labelFile, line);
            if(!line.size()) break;
            stringstream stream_line(line);
            while (stream_line.good()) {
                string substr;
                getline(stream_line, substr, ' ');
                labels[stoi(substr)].push_back({cnt++, embeds[i++]});
                if(i>=embeds.size()) {
                    break;
                }
            }
        }
    }

    for(auto label : labels) {
        outfile.open("batches/clusters/"+ to_string(label.first)+".txt", ios_base::app);

        auto embeds = label.second;
        for(auto embed : embeds) {
            outfile << embed.first << " ";
            for(auto element : embed.second) outfile << element << " ";
            outfile << endl;
        }
        outfile.close();
    }


}


int main(int argc, char **argv)
{

//    // generate points
//    const int npoints = 100;
//    vector<MyPoint> points;
//    ifstream file("embeds2.txt");
//    string writeFileName = "batches/0-embeds-batch.txt";
//    ofstream writeFile(writeFileName);
//
//
//    if (file.is_open()) {
//        int now = 0;
//        while ( file ) {
//            string line;
//            getline (file, line);
//            if(!line.size()) break;
//            line = line.substr(1, line.size() - 2);
//            stringstream stream_line(line);
//            vector<double> dims;
//            while (stream_line.good()) {
//                string substr;
//                getline(stream_line, substr, ',');
//                writeFile << substr << " ";
////                dims.push_back(stod(substr));
//            }
//            writeFile << endl;
//            now ++;
//            if(now == 100000) {
//                now = 0;
//                writeFileName[8]++;
//                writeFile.close();
//                writeFile.open(writeFileName);
//            }
////            points.push_back(MyPoint(dims));
//        }
//    }


    // build k-d tree
//    kdt::KDTree<MyPoint> kdtree(points);
    for (int i = 0; i < 5; ++i) {
        mapClusters(i);
    }

    return 0;
}