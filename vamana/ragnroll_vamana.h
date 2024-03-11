//
// Created by farah on 3/4/2024.
//

#ifndef INDEXER_RAGNROLL_VAMANA_H
#define INDEXER_RAGNROLL_VAMANA_H

#include <bits/stdc++.h>
#include <mutex>
#include "omp.h"

#define BLOCK_SIZE 3320
#define ROUND_UP(X, Y) \
((((uint64_t)(X) / (Y)) + ((uint64_t)(X) % (Y) != 0)) * (Y))

using namespace std;

class Vamana {
    typedef uint32_t idx_t;

public:
    struct cmp {
        constexpr bool operator() (std::pair<float, idx_t> const &a, std::pair<float, idx_t> const &b) const noexcept {
            return a.first < b.first;
        }
    };
    // Farah: this heap sorts based on highest distance
    // Question? why do we use max heap rather than min heap
    using maxHeap = std::priority_queue<std::pair<float, idx_t>, std::vector<std::pair<float, idx_t>>, cmp>;

    Vamana(size_t r, size_t l, float alp, size_t dim)
            : R(r), L(l), alpha(alp), dim(dim), link_list_locks(1000000) {
        data_size = dim * sizeof(float);
        link_size = R * sizeof(idx_t) + sizeof(idx_t);
        node_size = link_size + data_size;
        index_built = false;
        centroid = 0;
        nodes_no = 0;
    }

    Vamana() {

    }

    ~Vamana(){
        index_built = false;
    }

    void CreateIndex(vector<vector<float>>& pdata) {
        if (index_built) {
            std::cout << "Index is already built!!!!!!!!!!!" << std::endl;
            return;
        }
        nodes_no = pdata.size();
        graph.resize(nodes_no);
        points.resize(nodes_no);
        if (R < (size_t)std::ceil(log2(nodes_no))) {
            std::cout << "The parameter is less than log2(n), maybe result in low recall!!!!!!" << std::endl;
        }
        std::vector<std::mutex>(nodes_no).swap(link_list_locks);
        addPoints(pdata);
        randomInit();
        HealthyCheck();
        buildIndex(pdata);
        index_built = true;
    }

    vector<pair<float, idx_t>>
    Search(vector<float>& query, const size_t topk) {
        std::vector<std::pair<float, idx_t>> ret(topk);
        auto ans = search(query, centroid, topk);
        auto sz = ans.size();
        while (!ans.empty()) {
            sz --;
            ret[sz] = ans.top();
            ans.pop();
        }
        return ret;
    }

    void HealthyCheck() {
        std::vector<size_t> degree_hist;
        scan_graph(degree_hist);
        std::cout << "---show degree histogram of graph---" << std::endl;
        for (auto i = 0; i < degree_hist.size(); i ++) {
            std::cout << "degree = " << i << ": cnt = " << degree_hist[i] << std::endl;
        }
    }

    void writeIndexToFile(const std::string& path) {
        // disk index format
        /*
        first block: the number of point(size_t), dim(size_t), centroid_idx(size_t)

        4k per block: [vec: the number of neighbors: neighbors]
        */
        //std::ofstream fout(path, std::ios_base::binary);
        std::ofstream fout(path);
        // 4k per block
        char block[BLOCK_SIZE];
        int32_t neighbors[50];
        // write the number of point(size_t), dim(size_t), centroid_idx(size_t)
        int offset = 0;
        memset(block, -1, sizeof(block));

        memcpy(block, &nodes_no, sizeof(nodes_no));
        offset += sizeof(nodes_no);

        memcpy(block + offset, &dim, sizeof(dim));
        offset += sizeof(dim);

        memcpy(block + offset, &R, sizeof(R));
        offset += sizeof(R);

        memcpy(block + offset, &centroid, sizeof(centroid));

        fout.write(block, sizeof(char) * BLOCK_SIZE);

        size_t num_per_block =
                BLOCK_SIZE / (sizeof(float) * dim + sizeof(int32_t) * (R + 1));
        cout<<(sizeof(float) * dim + sizeof(int32_t) * (R + 1))<<endl;
        cout<<num_per_block;

        size_t block_num = ROUND_UP(nodes_no, num_per_block);
        size_t idx = 0;
        for (size_t block_id = 0; block_id < block_num; ++block_id) {
            memset(block, -1, sizeof(block));
            std::string raw_data;
            for (size_t id = 0; id < num_per_block and idx < nodes_no; ++id) {
                raw_data.append(reinterpret_cast<const char&>(points[idx]),
                                sizeof(float) * dim);
                int32_t num_neighbors = graph[idx].size();
                raw_data.append(reinterpret_cast<char*>(&num_neighbors), sizeof(int32_t));
                memset(neighbors, -1, sizeof(neighbors));
                size_t neighbors_idx = 0;
                for (auto& v : graph[idx]) {
                    neighbors[neighbors_idx++] = v;
                }
                raw_data.append(reinterpret_cast<char*>(neighbors),
                                sizeof(int32_t) * R);
                idx++;
            }
            memcpy(block, raw_data.data(), raw_data.size());
            fout.write(block, BLOCK_SIZE);
        }
        fout.close();
    }

private:
    void randomInit() {
        srand(unsigned(time(NULL)));
#pragma omp parallel
        {
            auto threads_no = omp_get_num_threads(); // number of threads
            auto curr_thread = omp_get_thread_num(); // thread number
            for (auto i = 0; i < nodes_no; i ++) {
                if (i % threads_no == curr_thread) {
                    std::set<idx_t> random_neighbors;
                    do {
                        random_neighbors.insert((idx_t)(rand() % nodes_no));
                    } while (random_neighbors.size() < R);
                    assert(random_neighbors.size() <= R);
                    for (auto &chosen : random_neighbors) {
                        // we need to case size_t type to idx_t
                        graph[(idx_t)i].push_back(chosen);
                    }
                }
            }
        }
    }

    void buildIndex(vector<vector<float>>& pdata) {
        assert(nodes_no > 0);
        vector<float> center(dim,0);

        for (size_t i = 0; i < nodes_no; i++) {
            for(size_t j = 0; j<dim;j++) center[j] += pdata[i][j];
        }
        for (auto i = 0; i < dim; i ++)
            center[i] /= nodes_no;
        auto tstart = std::chrono::high_resolution_clock::now();
        auto tpL = search(center, (idx_t)(rand() % nodes_no), L);
        auto tend = std::chrono::high_resolution_clock::now();
        std::cout << "first search for medoid finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms." << std::endl;
        while (!tpL.empty()) {
            centroid = tpL.top().second;
//            std::cout << "sp_ = " << sp_ << std::endl;
            tpL.pop();
        }
        std::cout << "init sp_ = " << centroid << std::endl;

        // step2: do the first iteration with alpha = 1
        tstart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100)
        for (idx_t i = 0; i < nodes_no; i ++) {
            maxHeap candidates;
            search(points[i], candidates);
            robustPrune(i, candidates, 1.0);
            make_edge(i, 1.0);
        }
        tend = std::chrono::high_resolution_clock::now();
        std::cout << "the first round iteration finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms." << std::endl;

        // todo: need update sp_?
        tpL = search(center, centroid, L);
        while (!tpL.empty()) {
            centroid = tpL.top().second;
            tpL.pop();
        }

        std::cout << "updated sp_ after 1st iteration: " << centroid << std::endl;

        std::cout << "HealthyCheck after the 1st round iteration:" << std::endl;
        HealthyCheck();

        // step3: do the second iteration with alpha = alpha
        tstart = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 100)
        for (int i = (int)(nodes_no - 1); i >= 0; i --) {
            maxHeap candidates;
            search(points[i], candidates);
            robustPrune(i, candidates, alpha);
            make_edge(i, alpha);
        }
        tend = std::chrono::high_resolution_clock::now();
        std::cout << "the second round iteration finished in " << std::chrono::duration_cast<std::chrono::milliseconds>(tend - tstart).count() << " ms." << std::endl;

        // step4: update sp_
        tpL = search(center, centroid, L);
        while (!tpL.empty()) {
            centroid = tpL.top().second;
            tpL.pop();
        }

        std::cout << "updated sp_ after 2nd iteration: " << centroid << std::endl;

        std::cout << "HealthyCheck after the 2nd round iteration:" << std::endl;
        HealthyCheck();
    }

    maxHeap search(vector<float>& query_point, maxHeap& neighbor_candi) {
        while (neighbor_candi.size()) {
            neighbor_candi.pop();
            std::cout << "neighbor_candi not empty" << std::endl;
        }
        std::vector<bool> vis(nodes_no, false);
        maxHeap resultSet;
        maxHeap expandSet;

        expandSet.emplace(-getDistance(points[centroid], query_point), centroid);

        vis[centroid] = true;
        float lowerBound = -expandSet.top().first;
        neighbor_candi.emplace(expandSet.top());

        while (expandSet.size()) {
            auto cur = expandSet.top();
            assert(cur.second < nodes_no);
            if ((-cur.first) > lowerBound)
                break;
            expandSet.pop();
            // getting adj of node
            auto link_size = graph[cur.second].size();
            if (link_size > R) {
                std::cout << "search: link size = " << link_size << " which is > R_ = " << R << std::endl;
            }
            assert(link_size <= R);
            std::unique_lock<std::mutex> lk(link_list_locks[cur.second]);
            for (auto i = 0; i < link_size; i ++) {
                auto candi_id = graph[cur.second][i];
                // todo: prefetch
                if (vis[candi_id])
                    continue;
                auto candi_data = points[cur.second];
                auto dist = getDistance(query_point, candi_data);
                if (resultSet.size() < L || dist < lowerBound) {
                    expandSet.emplace(-dist, candi_id);
                    vis[candi_id] = true;
                    neighbor_candi.emplace(-dist, candi_id);
                    resultSet.emplace(dist, candi_id);
                    if (resultSet.size() > L)
                        resultSet.pop();
                    if (!resultSet.empty())
                        lowerBound = resultSet.top().first;
                }
            }
        }
        return resultSet;
    }

    maxHeap search(vector<float>& query_point, const size_t centroid_, const size_t topk) {
        size_t upper_bound = L < topk ? topk : L;
        std::vector<bool> vis(nodes_no, false);
        maxHeap resultSet;
        maxHeap expandSet;
        expandSet.emplace(-getDistance(points[centroid_], query_point), centroid_);
        vis[centroid_] = true;
        float lowerBound = -expandSet.top().first;
        while (expandSet.size()) {
            auto cur = expandSet.top();
            assert(cur.second < nodes_no);
            if ((-cur.first) > lowerBound)
                break;
            expandSet.pop();
            auto link = graph[cur.second];
            auto link_size = link.size();
            if (link_size > R) {
                std::cout << "search_st: link_size = " << link_size << " which is > R = " << R << std::endl;
            }
            assert(link_size <= R);
            for (auto i = 0; i < link_size; i ++) {
                auto candi_id = link[i];
                // todo: prefetch
                if (vis[candi_id])
                    continue;
                auto candi_data = points[candi_id];
                auto dist = getDistance(query_point, candi_data);
                if (resultSet.size() < upper_bound || dist < lowerBound) {
                    expandSet.emplace(-dist, candi_id);
                    vis[candi_id] = true;
                    resultSet.emplace(dist, candi_id);
                    if (resultSet.size() > upper_bound)
                        resultSet.pop();
                    if (!resultSet.empty())
                        lowerBound = resultSet.top().first;
                }
            }
        }
        return resultSet;
    }

    void make_edge(const idx_t p, const float alpha) {
        for (auto i = 0; i < graph[p].size(); i ++) {
            idx_t neighbor_link = graph[p][i];
            std::unique_lock<std::mutex> lk(link_list_locks[neighbor_link]);
            if (!isDuplicate(p, graph[neighbor_link])) {
                if (graph[neighbor_link].size() < R) {
                    graph[neighbor_link].push_back(p);
                } else {
                    lk.unlock();
                    maxHeap pruneCandi;
                    auto dist = getDistance(points[p], points[neighbor_link]);
                    pruneCandi.emplace(-dist, p);
                    for (auto j = 0; j < graph[neighbor_link].size(); j ++) {
                        pruneCandi.emplace(-getDistance(points[neighbor_link],points[graph[neighbor_link][j]]), graph[neighbor_link][j]);
                    }
                    robustPrune(neighbor_link, pruneCandi, alpha);
                }
            }
        }
    }

    void robustPrune(const idx_t p, maxHeap& cand_set, const float alpha) {
        std::unique_lock<std::mutex> lock(link_list_locks[p]);
        if (cand_set.size() <= R) {
            graph[p].resize(cand_set.size());
            for (auto i = 0; i < cand_set.size(); i ++) {
                graph[p][i] = cand_set.top().second;
                cand_set.pop();
            }
            return;
        }
        while (cand_set.size() > 0) {
            if (graph[p].size() >= R)
                break;
            auto cur = cand_set.top();
            cand_set.pop();
            bool good = true;
            for (auto j = 0; j < graph[p].size(); j ++) {
                auto dist = getDistance(points[cur.second], points[graph[p][j]]);
                if (dist * alpha < -cur.first) {
                    good = false;
                    break;
                }
            }
            if (good) {
                graph[p].push_back(cur.second);
            }
        }
    }

    void addPoints(vector<vector<float>>& pdata) {
        for(int i=0;i<pdata.size();i++) {
            points[i] = pdata[i];
        }
    }

    float getDistance(vector<float>& point1, vector<float>& point2) {
        float dis = 0;
        for(int i=0;i<dim;i++) {
            float mid = point1[i] - point2[i];
            dis += mid*mid;
        }
        return dis;
    }

    bool isDuplicate(const idx_t p, vector<idx_t> adj) {
        assert(adj.size() <= R);
        for (auto i = 0; i < adj.size(); i++) {
            if (p == adj[i])
                return true;
        }
        return false;
    }

    void scan_graph(std::vector<size_t>& degree_histogram) {
        degree_histogram.resize(R + 1, 0);
        size_t total = 0;
#pragma omp parallel for reduction(+: total)
        for (auto i = 0; i < nodes_no; i ++) {
            auto link = graph[i];
            auto link_size = link.size();
            if (link_size > R) {
                std::cout << "scan_graph: *(link) = " << link_size << " which is > R_ = " << R << std::endl;
            }
            assert(link_size <= R);
#pragma omp critical
            {
                degree_histogram[link_size] ++;
            }
            std::set<idx_t> ns;
            for (auto j = 0; j < link_size; j ++) {
                assert(link[j] < nodes_no);
                ns.emplace(link[j]);
            }
            total += std::abs((int)(link_size) - (int)(ns.size()));
        }
        std::cout << "scan_graph done, duplicate total = " << total << std::endl;
    }


private:
    size_t L;
    size_t R;
    float alpha;
    size_t data_size;
    size_t link_size;
    size_t node_size;
    idx_t  centroid;
    vector<vector<idx_t>> graph;
    vector<vector<float>> points;
    bool index_built;
    size_t dim;
    size_t nodes_no;
    std::vector<std::mutex> link_list_locks;
};

#endif //INDEXER_RAGNROLL_VAMANA_H
