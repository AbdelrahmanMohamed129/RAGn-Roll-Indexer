//
// Created by farah on 3/4/2024.
//
#include <bits/stdc++.h>
#include "ragnroll_vamana.h"

using namespace std;

int main() {
    vector<pair<int,vector<float>>> pdata;
    const int dim = 768;
    int R = 12, L = 12, alpha = 1.2;
    auto x = new Vamana(R, L, alpha, dim);

    int topK = 10;

    for (int i = 0; i < 1; ++i) {
        ifstream myFile("./batches/clusters/" + to_string(i) + ".txt");
//        ifstream myFile("littleids.txt");
        if (myFile.is_open()) {
            while (myFile) {
                string line;
                getline(myFile, line);
                if (!line.size()) break;
//                line.erase(remove(line.begin(), line.end(), '['), line.end());
//                line.erase(remove(line.begin(), line.end(), ']'), line.end());
                stringstream stream_line(line);
                string id;
                getline(stream_line, id, ' ');
                if(!id.length()) continue;
                pdata.push_back({stoi(id),vector<float>()});
                while (stream_line.good()) {
                    string substr;
                    getline(stream_line, substr, ' ');
                    if (!substr.length()) continue;
                    pdata.back().second.push_back(stof(substr));
                }
//                cout << pdata.size() << " ";
            }
        }
    }

    auto tstart = std::chrono::high_resolution_clock::now();
    x->CreateIndex(pdata);
    auto tend = std::chrono::high_resolution_clock::now();
    std::cout << "create index done in " << std::chrono::duration_cast<std::chrono::milliseconds>( tend - tstart ).count() << " millseconds." << std::endl;

    // read the queries from the query.txt
    ifstream queryFile("query.txt");
    vector<vector<float>> queries;
    if (queryFile.is_open()) {
        while (queryFile) {
            string line;
            getline(queryFile, line);
            if (!line.size()) break;
            queries.push_back(vector<float>());
            stringstream stream_line(line);
            while (stream_line.good()) {
                string substr;
                getline(stream_line, substr, ' ');
                if (!substr.length()) continue;
                queries.back().push_back(stof(substr));
            }
        }
    }

    // calculate the ground truth of each query from the pdata and output the top 10 points where it outputs the distance then ID in a file each in a line
    for (int l = 0; l < queries.size(); ++l) {
        vector<pair<float, int>> groundTruth;
        for (int j = 0; j < pdata.size(); ++j) {
            float dist = 0;
            for (int k = 0; k < dim; ++k) {
                dist += (queries[l][k] - pdata[j].second[k]) * (queries[l][k] - pdata[j].second[k]);
            }
            groundTruth.push_back({dist, pdata[j].first});
        }
        sort(groundTruth.begin(), groundTruth.end());

//        ofstream groundTruthFile("groundtruth" + to_string(l) + ".txt");
//        for (int j = 0; j < topK; ++j) {
//            groundTruthFile << groundTruth[j].first << " " << groundTruth[j].second << endl;
//        }

        std::cout << "show groundtruth:" << std::endl;
        for (size_t j = 0; j < topK; j ++) {
            std::cout << "(" << groundTruth[j].second << ", " << groundTruth[j].first << ") ";
        }
        std::cout << std::endl;


        auto res = x->Search(queries[l], topK);
        std::cout << "show resultset:" << std::endl;
        for (size_t j = 0; j < topK; j ++) {
            std::cout << "(" << res[j].second << ", " << res[j].first << ") ";
        }
        std::cout << std::endl;


        // calculate the recall of the result set
        int recall = 0;
        for (int i = 0; i < topK; ++i) {
            for (int j = 0; j < topK; ++j) {
                if (res[i].second == groundTruth[j].second) {
                    recall++;
                    break;
                }
            }
        }
        cout << "Recall: " << (float)recall / topK << endl;

    }


//    x->writeIndexToFile("./indexed/test/0.txt");
    return 0;
}

