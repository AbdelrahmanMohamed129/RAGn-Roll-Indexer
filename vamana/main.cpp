//
// Created by farah on 3/4/2024.
//
#include <bits/stdc++.h>
#include "ragnroll_vamana.h"


using namespace std;

int numOfClusters = 100;
const int dim = 768;
int file_no = 10;
int total_files = 40;
int isImages = 0;
string headerPath = "Z:/";
// string headerPath = "D:/Boody/GP/Indexer/RAGn-Roll-Indexer/";


// ################################## Creating the Clusters ##################################
bool is_number(const std::string& s)
{
    return !s.empty() && std::find_if(s.begin(),
                                      s.end(), [](unsigned char c) { return !std::isdigit(c); }) == s.end();
}

void mapClustersFaiss() {

    vector<pair<int,vector<double>>> embeds;
    int cnt = 0;

    // Counting the number of files in the directory
    int batches(0);
    std::filesystem::path p1 { headerPath + "Data_"+ std::to_string(file_no) +"/normalizedDataset/" };
    for (auto& p : std::filesystem::directory_iterator(p1)) ++batches;

    cout<<"Number of Batches: "<<batches<<endl;

    int logger = 0;

    for(int batch = 0; batch < batches; batch++) {
        string batchFileName = headerPath + "Data_"+ std::to_string(file_no) +"/normalizedDataset/" + to_string(batch) + "-embeds-batch.txt";
        ifstream batchFile(batchFileName);

        if (batchFile.is_open()) {
            while (batchFile) {
                string line;
                getline(batchFile, line);
                if (!line.size()) break;
                stringstream stream_line(line);
                string id;
                getline(stream_line, id, ' ');
                if(!id.length()) continue;
                embeds.push_back({stoi(id),vector<double>()});
                while (stream_line.good()) {
                    string substr;
                    getline(stream_line, substr, ' ');
                    if (substr.length() < 2) continue;
                    embeds.back().second.push_back(stod(substr));
                }
                if(embeds.size() > logger) {
                    logger += 100000;
                    cout<<"Another 100000"<<endl;
                }
            }
        }
        cout << "DONE READING BATCH " << batch << "\n";
    }

    cout << "DONE READING EMBEDS\n";

    map<int,vector<pair<int, vector<double>>>> labels;
    string labelFileName = headerPath + "Data_"+ std::to_string(file_no) +"/labels/faiss_labels.txt";
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
                // In case we assign the same point to two clusters
                getline(stream_line, substr, ' ');
//                if(!substr.size()) break;
                if(!is_number(substr)) {
                    cout<<substr<<endl;
                    return;
                }
                labels[stoi(substr)].push_back({embeds[i].first, embeds[i].second});

                getline(stream_line, substr, ' ');
                if(!is_number(substr)) {
                    cout<<substr<<endl;
                    return;
                }
                labels[stoi(substr)].push_back({embeds[i].first, embeds[i++].second});
                if(i>=embeds.size()) {
                    break;
                }
            }
        }
    }

    cout << "DONE READING LABELS\n";

    for(auto label : labels) {
        outfile.open(headerPath + "Data_"+ std::to_string(file_no) +"/clusters/"+ to_string(label.first)+".txt", ios_base::app);

        auto embeds = label.second;
        for(auto embed : embeds) {
            outfile << embed.first << " ";
            for(auto element : embed.second) outfile << element << " ";
            outfile << endl;
        }
        outfile.close();
    }

    cout << "DONE MAPPING CLUSTERS\n";

    /* ---------------- Calculating the Number of Clusters ---------------- */
    numOfClusters = labels.size();
    cout<<"Number of Clusters: " << numOfClusters <<endl;
}


// ################################## Building the Index for All Clusters ##################################
void buildIndexForAllData(int R, int L, double alpha) {
    mapClustersFaiss();

    for (int i = 0; i < numOfClusters; ++i) {
        auto x = new Vamana(R, L, alpha, dim);
        vector<pair<int,vector<float>>> pdata;
        ifstream myFile(headerPath + "Data_"+ std::to_string(file_no) +"/clusters/" + to_string(i) + ".txt");
//        ifstream myFile("D:/Boody/GP/DataTrying/Data/clusters/0.txt");

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
        if (pdata.size() < L) {
            cout << "Cluster " << i << " has less than L points" << endl;
            continue;
        }
        /* ---------------- Creating the Index ---------------- */
        auto tstart = std::chrono::high_resolution_clock::now();
        x->CreateIndex(pdata);
        auto tend = std::chrono::high_resolution_clock::now();
        std::cout << "create index done in " << std::chrono::duration_cast<std::chrono::milliseconds>( tend - tstart ).count() << " millseconds." << std::endl;

        x->writeIndexToFileBoost(headerPath + "Data_"+ std::to_string(file_no) +"/indexed/test/" + to_string(i) + ".bin");
//        x->writeIndexToFileBoost("D:/Boody/GP/DataTrying/Data/indexed/0.txt");
    }
}

vector<vector<double>> loadCentroids() {
    vector<vector<double>> centroids;
    ifstream myFile(headerPath + "Data_"+ std::to_string(file_no) +"/centroids/final-centroids.txt");
    if (myFile.is_open()) {
        while (myFile) {
            string line;
            getline(myFile, line);
            if (!line.size()) break;
            stringstream stream_line(line);
            centroids.push_back(vector<double>());
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


int main(int argc, char** argv) {
    // calculate time of execution
    auto start = std::chrono::high_resolution_clock::now();
    int R = 20; int L = 30; double alpha = 1; int topK = 10; bool buildIndex = false;

    // Read top K and headerFile only if we're retrieving for a query
    if (argc == 6) {
        topK = stoi(argv[1]);
        cout<<"topK: "<<topK<<endl;
        headerPath = argv[2];
        cout<<"headerPath: "<<headerPath<<endl;
        file_no = stoi(argv[3]);
        cout<<"file_no: "<<file_no<<endl;
    }
        // Read R, L, alpha, topK from the argv if we're building the index
    else if (argc > 3) {
        R = stoi(argv[1]);
        cout<<"R: "<<R<<endl;
        L = stoi(argv[2]);
        cout<<"L: "<<L<<endl;
        alpha = stof(argv[3]);
        cout<<"alpha: "<<alpha<<endl;
        topK = stoi(argv[4]);
        cout<<"topK: "<<topK<<endl;
        buildIndex = stoi(argv[5]);
        cout<<"buildIndex: "<<buildIndex<<endl;
        headerPath = argv[6];
        cout<<"headerPath: "<<headerPath<<endl;
        file_no = stoi(argv[7]);
        cout<<"file_no: "<<file_no<<endl;
        isImages = stoi(argv[8]);
        cout << "isImages: " << isImages << endl;
//        headerPath = isImages ? "Z:/DataImgs/" : "Z:/Data/";
    }

    /* ---------------- Building the Index ---------------- */
    if(buildIndex) buildIndexForAllData(R, L, alpha);


    /* ---------------- Reading the Query File ---------------- */
    // read the queries from the query.txt
    ifstream queryFile( headerPath + "query.txt");
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
    // normalize the query vectors
    for (int i = 0; i < queries.size(); ++i) {
        double sum = 0;
        for (int j = 0; j < queries[i].size(); ++j) {
            sum += queries[i][j] * queries[i][j];
        }
        sum = sqrt(sum);
        for (int j = 0; j < queries[i].size(); ++j) {
            queries[i][j] /= sum;
        }
    }


    /* ---------------- Getting Nearest nprobes Clusters from FAISS ---------------- */
    vector<vector<int>> nearestClusters;
    for (int i = 0; i < queries.size(); ++i) {
        // read the file query_faiss_labels and get the nearest clusters
        ifstream queryFaissFile(headerPath + "Data_"+ std::to_string(file_no) +"/labels/query_faiss_labels.txt");
        vector<int> faissLabels;
        if (queryFaissFile.is_open()) {
            while (queryFaissFile) {
                string line;
                getline(queryFaissFile, line);
                if (!line.size()) break;
                stringstream stream_line(line);
                while (stream_line.good()) {
                    string substr;
                    getline(stream_line, substr, ' ');
                    if (!substr.length()) continue;
                    faissLabels.push_back(stoi(substr));
                }
            }
        }
        nearestClusters.push_back(faissLabels);
    }


/* ---------------- Loading the Index ---------------- */
    for (int l = 0; l < queries.size(); ++l) {

        /* ---------------- Calculating the Ground-truth ---------------- */
    set<pair<double, int>> groundTruth;
//    for(int i = 0; i < numOfClusters; i++) {
//        if(!std::filesystem::exists(headerPath + "Data_"+ std::to_string(file_no) +"/indexed/test/" + to_string(i) + ".bin")) continue;
//        Vamana* loadedIndex = new Vamana();
//        loadedIndex->readIndexFromFileBoost(headerPath + "Data_"+ std::to_string(file_no) +"/indexed/test/" + to_string(i) + ".bin");
//
//        auto pdata = loadedIndex->getPoints();
//        auto ids = loadedIndex->getActualIds();
//
//        for (int j = 0; j < pdata.size(); ++j) {
//            double dist = 0;
//            dist = Vamana::getDistance(queries[l], pdata[j]);
//            groundTruth.insert({dist, ids[j]});
//        }
//    }

        vector<pair<float, uint32_t>> res;
        map<uint32_t, bool> vis;

//            for (int probe = 0; probe < nearestClusters[l].size(); probe++) {
#pragma omp parallel for
        for (int probe = 0; probe < 2; probe++) {

            if (!std::filesystem::exists(
                headerPath + "Data_" + std::to_string(file_no) + "/indexed/test/" +
                to_string(nearestClusters[l][probe]) + ".bin"))
            continue;
            Vamana *loadedIndex = new Vamana();

            loadedIndex->readIndexFromFileBoost(
                    headerPath + "Data_" + std::to_string(file_no) + "/indexed/test/" +
                    to_string(nearestClusters[l][probe]) + ".bin");
            vector<vector<float>> pdata = loadedIndex->getPoints();

            loadedIndex->Search(queries[l], topK, res, vis);

            // Searching for the query (Replaced with reference variable in Search of the Vamana class)
            //            vector<pair<float,uint32_t>> tempRes = loadedIndex->Search(queries[l], topK);
            //            for (int j = 0; j < topK; ++j) {
            //                if(vis[tempRes[j].second]) continue;
            //                vis[tempRes[j].second] = true;
            //                res.push_back({tempRes[j].first, tempRes[j].second});
            //            }
        }

        sort(res.begin(), res.end());

        /* ---------------- Showing the Ground-truth and Calculating Recall ---------------- */

        // calculate the ground truth of each query from the pdata and output the top 10 points where it outputs the distance then ID in a file each in a line
        int remK(topK);

//        std::cout << "show groundtruth:" << std::endl;
//        for (auto j : groundTruth) {
//            if (remK == 0) break;
//            remK--;
//            std::cout << "(" << j.second << ", " << j.first << ") ";
//        }
//        std::cout << std::endl;


//         std::cout << "show resultset:" << std::endl;
//         remK = topK;
//         for (auto j: res) {
//             if (remK == 0) break;
//             remK--;
//             std::cout << "(" << j.second << ", " << j.first << ") ";
//         }
//         std::cout << std::endl;


        // calculate the recall of the result set
//        int recall = 0;
//        remK = topK;
//        for (auto i : groundTruth) {
//            if (remK == 0) break;
//            remK--;
//            int remKRes(topK);
//            for (auto j : res) {
//                if (remKRes == 0) break;
//                remKRes--;
//                if (i.second == j.second) {
//                    recall++;
//                    break;
//                }
//            }
//        }
//        cout << "Recall: " << (double)recall / topK << endl;
        auto end = std::chrono::high_resolution_clock::now();

//         std::cout << "Retrieval done in "
//                   << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " millseconds."
//                   << std::endl;

        // write the result set in a file
        remK = topK;
        ofstream outfile(headerPath + "Data_" + std::to_string(file_no) + "/res.txt");
        for (auto j: res) {
            if (remK == 0) break;
            remK--;
            outfile << j.second << " " << -j.first << endl;
        }
        outfile.close();

        // write the ground truth in a file
//        ofstream outfile2(headerPath + "Data_"+ std::to_string(file_no) +"/gt.txt");
//        remK = topK;
//        for (auto j : groundTruth) {
//            if (remK == 0) break;
//            remK--;
//            outfile2 << j.second << " " << -j.first << endl;
//        }
//        outfile2.close();
    }


    return 0;
}

/* ---------------- Getting Nearest nprobes Clusters by calculating our distances ---------------- */
//    vector<vector<int>> nearestClusters;
//    for (int i = 0; i < queries.size(); ++i) {
//        vector<pair<double, int>> dists;
//        for (int j = 0; j < centroids.size(); ++j) {
//            double dist = 0;
////            for (int k = 0; k < dim; ++k) {
////                dist += (queries[i][k] - centroids[j][k]) * (queries[i][k] - centroids[j][k]);
////            }
//            dist = Vamana::getDistance(queries[i], centroids[j]);
//            dists.push_back({dist, j});
//        }
//        sort(dists.begin(), dists.end());
//        vector<int> nearest;
//        for (int j = 0; j < nprobes; ++j) {
//            nearest.push_back(dists[j].second);
//        }
//        nearestClusters.push_back(nearest);
//    }

