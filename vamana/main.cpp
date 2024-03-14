//
// Created by farah on 3/4/2024.
//
#include <bits/stdc++.h>
#include "ragnroll_vamana.h"

using namespace std;

int main() {
    vector<pair<int,vector<float>>> pdata;
    const int dim = 768;
    auto x = new Vamana(10, 10, 1.5, dim);
//    for (int i = 0; i < 100; ++i) {
//        ifstream myFile("./batches/clusters/" + to_string(i) + ".txt");
        ifstream myFile("littleids.txt");
        if (myFile.is_open()) {
            while (myFile) {
                string line;
                getline(myFile, line);
                if (!line.size()) break;
                line.erase(remove(line.begin(), line.end(), '['), line.end());
                line.erase(remove(line.begin(), line.end(), ']'), line.end());
                stringstream stream_line(line);
                string id;
                getline(stream_line, id, ' ');
                pdata.push_back({stoi(id),vector<float>()});
                while (stream_line.good()) {
                    string substr;
                    getline(stream_line, substr, ' ');
                    if (!substr.length()) continue;
                    pdata.back().second.push_back(stof(substr));
                }
            }
        }
//    }

//    // read the queries from the query.txt
//    ifstream queryFile("query.txt");
//    vector<vector<float>> queries;
//    if (queryFile.is_open()) {
//        while (queryFile) {
//            string line;
//            getline(queryFile, line);
//            if (!line.size()) break;
//            queries.push_back(vector<float>());
//            stringstream stream_line(line);
//            while (stream_line.good()) {
//                string substr;
//                getline(stream_line, substr, ' ');
//                if (!substr.length()) continue;
//                queries.back().push_back(stof(substr));
//            }
//        }
//    }
//    // calculate the ground truth of each query from the pdata and output the top 10 points where it outputs the distance then ID in a file each in a line
//    for (int l = 0; l < queries.size(); ++l) {
//        vector<pair<float, int>> groundTruth;
//        for (int j = 0; j < pdata.size(); ++j) {
//            float dist = 0;
//            for (int k = 0; k < dim; ++k) {
//                dist += (queries[l][k] - pdata[j].second[k]) * (queries[l][k] - pdata[l].second[k]);
//            }
//            groundTruth.push_back({dist, j});
//        }
//        sort(groundTruth.begin(), groundTruth.end());
//        ofstream groundTruthFile("groundtruth" + to_string(l) + ".txt");
//        for (int j = 0; j < 20; ++j) {
//            groundTruthFile << groundTruth[j].first << " " << groundTruth[j].second << endl;
//        }
//    }


    x->CreateIndex(pdata);
//
//    // Generate 3 random query of dim 768 and make it from -1 to 1
//    vector<float> query;
//    for (int i = 0; i < dim; ++i) {
//        query.push_back((float) rand() / RAND_MAX * 2 - 1);
//    }
//
//    // calculate the ground truth of this query from the pdata
//    vector<pair<float, int>> groundTruth;
//    for (int i = 0; i < pdata.size(); ++i) {
//        float dist = 0;
//        for (int j = 0; j < dim; ++j) {
//            dist += (query[j] - pdata[i][j]) * (query[j] - pdata[i][j]);
//        }
//        groundTruth.push_back({dist, i});
//    }
//    sort(groundTruth.begin(), groundTruth.end());
//
//    auto res = x->Search(query, 10);
//
//    // print the L2 distance of the res
//    for (int i = 0; i < 10; ++i) {
//        cout << res[i].first << " ";
//    }
//    cout << endl;
//
//    // print the ground truth
//    for (int i = 0; i < 10; ++i) {
//        cout << groundTruth[i].first << " ";
//    }
//    cout << endl;
//
//    // evaluate the recall of res along with the ground truth get the top 10 results only
//    int recall = 0;
//    for (int i = 0; i < 10; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            if (res[i].second == groundTruth[j].second) {
//                recall++;
//                break;
//            }
//        }
//    }
//    cout << "Recall: " << recall / 10.0 << endl;


//    // generate 3 random query of dim 768 and make it from -1 to 1 and output them in a file each in a line
//    for (int i = 0; i < 3; ++i) {
//        ofstream queryFile(to_string(i) + ".txt");
//        for (int j = 0; j < dim; ++j) {
//            queryFile << (float) rand() / RAND_MAX * 2 - 1 << " ";
//        }
//        queryFile << endl;
//    }

//    x->writeIndexToFile("./indexed/test/0.txt");
    return 0;
}

