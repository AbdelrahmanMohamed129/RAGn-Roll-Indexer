//
// Created by farah on 3/4/2024.
//
#include <bits/stdc++.h>
#include "ragnroll_vamana.h"

using namespace std;

void buildIndexForAllData() {
    const int dim = 768;
    int R = 20, L = 30, alpha = 1.5;
    int numOfFiles = 100;

    for (int i = 0; i < numOfFiles; ++i) {
        auto x = new Vamana(R, L, alpha, dim);
        vector<pair<int,vector<float>>> pdata;
        ifstream myFile("./batches/clusters/" + to_string(i) + ".txt");
//        ifstream myFile("cluster0.txt");
        if (myFile.is_open()) {
            while (myFile) {
                string line;
                getline(myFile, line);
                if (!line.size()) break;
                // Removing square brackets
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

        /* ---------------- Creating the Index ---------------- */
        auto tstart = std::chrono::high_resolution_clock::now();
        x->CreateIndex(pdata);
        auto tend = std::chrono::high_resolution_clock::now();
        std::cout << "create index done in " << std::chrono::duration_cast<std::chrono::milliseconds>( tend - tstart ).count() << " millseconds." << std::endl;

        x->writeIndexToFileBoost("./batches/indexed/test/" + to_string(i) + ".bin");
    }
}

vector<vector<int>> loadCentroids() {
    vector<vector<int>> centroids;
    ifstream myFile("./batches/centroids/final-centroids.txt");
    if (myFile.is_open()) {
        while (myFile) {
            string line;
            getline(myFile, line);
            if (!line.size()) break;
            stringstream stream_line(line);
            centroids.push_back(vector<int>());
            while (stream_line.good()) {
                string substr;
                getline(stream_line, substr, ' ');
                if (!substr.length()) continue;
                centroids.back().push_back(stoi(substr));
            }
        }
    }
    return centroids;
}

int main() {
    const int dim = 768;
    int topK = 10;
    int nprobes = 23;

    /* ---------------- Building the Index ---------------- */
    //buildIndexForAllData();

    /* ---------------- Reading the Centroids ---------------- */
    auto centroids = loadCentroids();


    /* ---------------- Reading the Query File ---------------- */
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

    /* ---------------- Getting Nearest nprobes Clusters ---------------- */
    vector<vector<int>> nearestClusters;
    for (int i = 0; i < queries.size(); ++i) {
        vector<pair<float, int>> dists;
        for (int j = 0; j < centroids.size(); ++j) {
            float dist = 0;
            for (int k = 0; k < dim; ++k) {
                dist += (queries[i][k] - centroids[j][k]) * (queries[i][k] - centroids[j][k]);
            }
            dists.push_back({dist, j});
        }
        sort(dists.begin(), dists.end());
        vector<int> nearest;
        for (int j = 0; j < nprobes; ++j) {
            nearest.push_back(dists[j].second);
        }
        nearestClusters.push_back(nearest);
    }

    /* ---------------- Loading the Index ---------------- */
    for (int l = 0; l < queries.size(); ++l) {
        set<pair<float, int>> groundTruth;
        for(int i=0;i<100;i++) {
            Vamana* loadedIndex = new Vamana();
            loadedIndex->readIndexFromFileBoost("./batches/indexed/test/" + to_string(i) + ".bin");

            auto pdata = loadedIndex->getPoints();
            auto ids = loadedIndex->getActualIds();

            for (int j = 0; j < pdata.size(); ++j) {
                float dist = 0;
                for (int k = 0; k < dim; ++k) {
                    dist += (queries[l][k] - pdata[j][k]) * (queries[l][k] - pdata[j][k]);
                }
                groundTruth.insert({dist, ids[j]});
            }
        }
        set<pair<float, int>> res;
        for(int probe = 0; probe < nprobes; probe++) {
            Vamana* loadedIndex = new Vamana();
            loadedIndex->readIndexFromFileBoost("./batches/indexed/test/" + to_string(nearestClusters[l][probe]) + ".bin");

            auto pdata = loadedIndex->getPoints();

            // Searching for the query
            vector<pair<float,uint32_t>> tempRes = loadedIndex->Search(queries[l], topK);
            for (int j = 0; j < topK; ++j) {
                res.insert({tempRes[j].first, tempRes[j].second});
            }
        }

        /* ---------------- Calculating the Ground-truth and Recall ---------------- */

        // calculate the ground truth of each query from the pdata and output the top 10 points where it outputs the distance then ID in a file each in a line

        std::cout << "show groundtruth:" << std::endl;
        int remK(topK);
        for (auto j : groundTruth) {
            if (remK == 0) break;
            remK--;
            std::cout << "(" << j.second << ", " << j.first << ") ";
        }
        std::cout << std::endl;


        std::cout << "show resultset:" << std::endl;
        remK = topK;
        for (auto j : res) {
            if (remK == 0) break;
            remK--;
            std::cout << "(" << j.second << ", " << j.first << ") ";
        }
        std::cout << std::endl;

        // calculate the recall of the result set
        int recall = 0;
        remK = topK;
        for (auto i : groundTruth) {
            if (remK == 0) break;
            remK--;
            int remKRes(topK);
            for (auto j : res) {
                if (remKRes == 0) break;
                remKRes--;
                if (i.second == j.second) {
                    recall++;
                    break;
                }
            }
        }
        cout << "Recall: " << (float)recall / topK << endl;
    }

    return 0;
}

