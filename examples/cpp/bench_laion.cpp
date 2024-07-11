#include <omp.h>

#include <chrono>
#include <thread>

#include "../../hnswlib/hnswlib.h"

float ComputeRecall(uint32_t q_num, uint32_t k, uint32_t gt_dim, uint32_t *res, uint32_t *gt) {
    uint32_t total_count = 0;
    for (uint32_t i = 0; i < q_num; i++) {
        std::vector<uint32_t> one_gt(gt + i * gt_dim, gt + i * gt_dim + k);
        std::vector<uint32_t> intersection;
        std::vector<uint32_t> temp_res(res + i * k, res + i * k + k);
        // check if there duplication in temp_res
        // std::sort(temp_res.begin(), temp_res.end());

        // std::sort(one_gt.begin(), one_gt.end());
        for (auto p : one_gt) {
            if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end()) intersection.push_back(p);
        }
        // std::set_intersection(temp_res.begin(), temp_res.end(), one_gt.begin(), one_gt.end(),
        //   std::back_inserter(intersection));

        total_count += static_cast<uint32_t>(intersection.size());
    }
    return static_cast<float>(total_count) / (k * q_num);
    // return static_cast<float>(total_count) / (k * test_connected_q.size());
}

template <typename T>
void load_gt_meta(const char *filename, unsigned &points_num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&points_num, 4);
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    uint32_t calc_contained_pts = (unsigned)((fsize - sizeof(uint32_t) * 2) / (dim) / sizeof(T));
    if (points_num * 2 != calc_contained_pts) {
        std::cerr << "filename: " << std::string(filename) << std::endl;
        std::cerr << "Data file size wrong! Get points " << calc_contained_pts << " but should have " << points_num
                  << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}
template <typename T>
void load_meta(const char *filename, unsigned &points_num, unsigned &dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.read((char *)&points_num, 4);
    in.read((char *)&dim, 4);
    in.seekg(0, std::ios::end);
    std::ios::pos_type ss = in.tellg();
    size_t fsize = (size_t)ss;
    uint32_t calc_contained_pts = (unsigned)((fsize - sizeof(uint32_t) * 2) / (dim) / sizeof(T));
    std::cout << "load meta from file: " << filename << " points_num: " << points_num << " dim: " << dim << std::endl;
    if (points_num != calc_contained_pts) {
        std::cerr << "filename: " << std::string(filename) << std::endl;
        std::cerr << "Data file size wrong! Get points " << calc_contained_pts << " but should have " << points_num
                  << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}

// load bin data from file with different data type, so use template
template <typename T>
void load_gt_data(const char *filename, uint32_t &points_num, uint32_t &dim, T *&data) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.seekg(sizeof(uint32_t) * 2, std::ios::beg);
    data = new T[points_num * dim];
    for (size_t i = 0; i < points_num; i++) {
        // in.seekg(8 + i * (dim + 1) * sizeof(T), std::ios::beg);
        in.read((char *)(data + i * dim), dim * sizeof(T));
    }
    // cursor position
    std::ios::pos_type ss = in.tellg();
    if ((size_t)ss != points_num * dim * sizeof(T) + sizeof(uint32_t) * 2) {
        std::cerr << "Read file incompleted!" << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    in.close();
}

template <typename T>
void load_data(const char *filename, uint32_t &points_num, uint32_t &dim, T *&data) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cout << "open file error" << std::endl;
        exit(-1);
    }
    in.seekg(sizeof(uint32_t) * 2, std::ios::beg);
    data = new T[points_num * dim];
    for (size_t i = 0; i < points_num; i++) {
        // in.seekg(8 + i * (dim + 1) * sizeof(T), std::ios::beg);
        in.read((char *)(data + i * dim), dim * sizeof(T));
    }
    // cursor position
    std::ios::pos_type ss = in.tellg();
    if ((size_t)ss != points_num * dim * sizeof(T) + sizeof(uint32_t) * 2) {
        std::cerr << "Read file incompleted! filename:" << std::string(filename) << std::endl;
        throw std::runtime_error("Data file size wrong!");
    }
    std::cout << "load data from file: " << filename << " points_num: " << points_num << " dim: " << dim << std::endl;
    in.close();
}

template <typename T>
inline void normalize(T *arr, const size_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; i++) {
        sum += arr[i] * arr[i];
    }
    sum = sqrt(sum);
    for (uint32_t i = 0; i < dim; i++) {
        arr[i] = (T)(arr[i] / sum);
    }
}

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

// void normalize_vector(float* data, float* norm_array, uint32_t dim) {
//     float norm = 0.0f;
//     for (int i = 0; i < dim; i++)
//         norm += data[i] * data[i];
//     norm = 1.0f / (sqrtf(norm) + 1e-30f);
//     for (int i = 0; i < dim; i++)
//         norm_array[i] = data[i] * norm;
// }

// template<typename T>
// inline void normalize(T* arr, const size_t dim) {
//   float sum = 0.0f;
//   for (uint32_t i = 0; i < dim; i++) {
//     sum += arr[i] * arr[i];
//   }
//   sum = sqrt(sum);
//   for (uint32_t i = 0; i < dim; i++) {
//     arr[i] = (T)(arr[i] / sum);
//   }
// }

int main() {
    std::string base_file;
    std::string gt_file;
    std::string query_file;
    base_file = "/ann/dataset/laion/laion400M/img_emb_0.fbin";
    gt_file = "/ann/dataset/laion/laion400M/groundtruth.img0.1M.text0.1M.ibin";
    query_file = "/ann/dataset/laion/laion400M/text_emb_0.2k.fbin";
    uint32_t points_num, dim;
    float *base_data, *query_data;
    uint32_t *gt_data;
    load_meta<float>(base_file.c_str(), points_num, dim);
    load_data<float>(base_file.c_str(), points_num, dim, base_data);
    uint32_t gt_num, gt_dim;
    load_gt_meta<uint32_t>(gt_file.c_str(), gt_num, gt_dim);
    load_gt_data<uint32_t>(gt_file.c_str(), gt_num, gt_dim, gt_data);
    uint32_t query_num, query_dim;
    load_meta<float>(query_file.c_str(), query_num, query_dim);
    load_data<float>(query_file.c_str(), query_num, query_dim, query_data);

    // int dim = 16;               // Dimension of the elements
    int max_elements = points_num;  // Maximum number of elements, should be known beforehand
    int M = 32;                     // Tightly connected with internal dimensionality of the data
                                    // strongly affects the memory consumption

    int ef_construction = 500;  // Controls index search speed/build speed tradeoff
    int num_threads = 90;       // Number of threads for operations with index

    // Initing index
    // hnswlib::L2Space space(dim);
    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float> *alg_hnsw =
        new hnswlib::HierarchicalNSW<float>(&space, points_num, M, ef_construction);

    // Generate random data
    // std::mt19937 rng;
    // rng.seed(47);
    // std::uniform_real_distribution<> distrib_real;
    // float* data = new float[dim * max_elements];
    // for (int i = 0; i < dim * max_elements; i++) {
    //     data[i] = distrib_real(rng);
    // }
    for (int i = 0; i < points_num; i++) {
        normalize<float>(base_data + dim * i, dim);
    }
    // Add data to index
    // ParallelFor(0, points_num, num_threads, [&](size_t row, size_t threadId) {
    //     // only for cosine now!!!! because always normalize
    //     alg_hnsw->addPoint((void*)(base_data + dim * row), row);
    // });
    // alg_hnsw->saveIndex("laion_1M_hnsw.index");
    alg_hnsw->loadIndex("laion_1M_hnsw.index", &space, points_num);
    uint32_t k = 10;
    // Query the elements for themselves and measure recall
    std::vector<uint32_t> neighbors(points_num * k);
    for (size_t i = 0; i < query_num; ++i) {
        normalize<float>(query_data + i * query_dim, dim);
    }
    std::vector<uint32_t> ef_vec = {10,  15,  20,  25,   30,   35,   40,   45,   50,   55,   60,   65,   70,   75,
                                    80,  85,  90,  95,   100,  110,  120,  130,  140,  150,  160,  170,  180,  190,
                                    200, 220, 240, 260,  280,  300,  350,  400,  450,  500,  550,  600,  650,  700,
                                    750, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000};
    omp_set_num_threads(1);
    for (auto ef : ef_vec) {
        alg_hnsw->setEf(ef);
        auto s = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1)
        for (uint32_t row = 0; row < query_num; ++row) {
            std::priority_queue<std::pair<float, hnswlib::labeltype>> result =
                alg_hnsw->searchKnn(query_data + dim * row, k);
            for (size_t i = 0; i < k; ++i) {
                hnswlib::labeltype label = result.top().second;
                neighbors[row * k + i] = static_cast<uint32_t>(label);
                result.pop();
            }
        }
        auto e = std::chrono::high_resolution_clock::now();

        float qps = static_cast<float>(query_num) /
                    (float)std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count() * 1000.0;
        // ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        //     std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row,
        //     1); hnswlib::labeltype label = result.top().second; neighbors[row] = label;
        // });
        float recall = ComputeRecall(query_num, k, gt_dim, neighbors.data(), gt_data);
        std::cout << "ef: " << ef << "\tQPS: " << qps << "\tRecall: " << recall << "\n";
    }
    delete[] base_data;
    delete alg_hnsw;
    return 0;
}
