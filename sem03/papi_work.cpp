#include <iostream>
#include <vector>
#include <cstdio>
#include <cstring>
#include <papi.h>

#define N_TESTS 5

// 定义CSR图结构
class CSR_graph {
    int row_count; // 图中顶点的数量
    unsigned int col_count; // 图中边的数量

    std::vector<unsigned int> row_ptr; // 行指针数组，存储每个顶点起始边的索引
    std::vector<int> col_ids; // 列索引数组，存储每条边指向的目标顶点
    std::vector<double> vals; // 边权重数组

public:
    // 从文件中读取CSR图数据
    void read(const char* filename) {
        FILE *graph_file = fopen(filename, "rb");
        if (!graph_file) {
            std::cerr << "Error opening file: " << filename << std::endl;
            exit(1);
        }

        if (fread(&row_count, sizeof(int), 1, graph_file) != 1) {
            std::cerr << "Error reading row_count from file: " << filename << std::endl;
            fclose(graph_file);
            exit(1);
        }
        if (fread(&col_count, sizeof(unsigned int), 1, graph_file) != 1) {
            std::cerr << "Error reading col_count from file: " << filename << std::endl;
            fclose(graph_file);
            exit(1);
        }

        std::cout << "Row_count = " << row_count << ", col_count = " << col_count << std::endl;

        row_ptr.resize(row_count + 1);
        col_ids.resize(col_count);
        vals.resize(col_count);

        if (fread(row_ptr.data(), sizeof(unsigned int), row_count + 1, graph_file) != row_count + 1) {
            std::cerr << "Error reading row_ptr from file: " << filename << std::endl;
            fclose(graph_file);
            exit(1);
        }
        if (fread(col_ids.data(), sizeof(int), col_count, graph_file) != col_count) {
            std::cerr << "Error reading col_ids from file: " << filename << std::endl;
            fclose(graph_file);
            exit(1);
        }
        if (fread(vals.data(), sizeof(double), col_count, graph_file) != col_count) {
            std::cerr << "Error reading vals from file: " << filename << std::endl;
            fclose(graph_file);
            exit(1);
        }
        fclose(graph_file);
    }

    // 重置图数据
    void reset() {
        row_count = 0;
        col_count = 0;
        row_ptr.clear();
        col_ids.clear();
        vals.clear();
    }

    // 获取顶点数量
    int get_vertex_count() const {
        return row_count;
    }

    // 获取指定顶点的所有边及其权重
    std::vector<std::pair<int, double>> get_edges(int vertex) const {
        std::vector<std::pair<int, double>> edges;
        if (vertex < 0 || vertex >= row_count) {
            std::cerr << "Vertex index out of bounds: " << vertex << std::endl;
            return edges;
        }
        for (unsigned int col = row_ptr[vertex]; col < row_ptr[vertex + 1]; ++col) {
            edges.emplace_back(col_ids[col], vals[col]);
        }
        return edges;
    }
};

// 实现第一个算法：找到具有最大总权重（指向偶数顶点）的顶点
int find_max_weight_vertex(const CSR_graph &graph) {
    int max_vertex = -1;
    double max_weight = -1.0; // 初始化为负值，以便处理所有非负权重

    for (int i = 0; i < graph.get_vertex_count(); ++i) {
        double total_weight = 0.0;
        auto edges = graph.get_edges(i);
        if (edges.empty()) {
            continue;
        }
        for (const auto &edge : edges) {
            if (edge.first % 2 == 0) { // 只计算指向偶数顶点的边
                total_weight += edge.second;
            }
        }
        if (total_weight > max_weight) {
            max_weight = total_weight;
            max_vertex = i;
        }
    }

    return max_vertex;
}

// 实现第二个算法：找到具有最大Rank的顶点
int find_max_rank_vertex(const CSR_graph &graph) {
    int max_vertex = -1;
    double max_rank = -1.0; // 初始化为负值，以便处理所有非负权重

    for (int i = 0; i < graph.get_vertex_count(); ++i) {
        double rank = 0.0;
        auto edges = graph.get_edges(i);
        if (edges.empty()) {
            continue;
        }
        for (const auto &edge : edges) {
            int target_vertex = edge.first;
            double weight = edge.second;
            double vertex_weight = 0.0;

            // 计算目标顶点的出边权重之和 W(j)
            auto target_edges = graph.get_edges(target_vertex);
            if (target_edges.empty()) {
                continue;
            }
            for (const auto &target_edge : target_edges) {
                vertex_weight += target_edge.second;
            }
            rank += weight * vertex_weight;
        }
        if (rank > max_rank) {
            max_rank = rank;
            max_vertex = i;
        }
    }

    return max_vertex;
}

// 使用PAPI监控单个算法的性能
void monitor_performance(const CSR_graph &graph, int &max_vertex, int algorithm_index, long long *values) {
    int event_set = PAPI_NULL;
    int retval;

    // 创建事件集并添加PAPI事件
    retval = PAPI_create_eventset(&event_set);
    if (retval != PAPI_OK) {
        std::cerr << "PAPI create event set error: " << PAPI_strerror(retval) << std::endl;
        exit(1);
    }

    int events[7] = {PAPI_L1_DCM, PAPI_L1_ICM, PAPI_L1_TCM, PAPI_L2_DCM, PAPI_L2_ICM, PAPI_L2_TCM, PAPI_TOT_INS};
    const char* event_names[7] = {"PAPI_L1_DCM", "PAPI_L1_ICM", "PAPI_L1_TCM", "PAPI_L2_DCM", "PAPI_L2_ICM", "PAPI_L2_TCM", "PAPI_TOT_INS"};

    for (int i = 0; i < 7; ++i) {
        retval = PAPI_query_event(events[i]);
        if (retval != PAPI_OK) {
            std::cerr << "PAPI event " << event_names[i] << " not supported: " << PAPI_strerror(retval) << std::endl;
            exit(1);
        }
        retval = PAPI_add_event(event_set, events[i]);
        if (retval != PAPI_OK) {
            std::cerr << "PAPI add event " << event_names[i] << " error: " << PAPI_strerror(retval) << std::endl;
            exit(1);
        }
    }

    // 清零 values 数组
    memset(values, 0, sizeof(long long) * 7);

    // 性能监测
    retval = PAPI_start(event_set);
    if (retval != PAPI_OK) {
        std::cerr << "PAPI start counters error: " << PAPI_strerror(retval) << std::endl;
        exit(1);
    }

    // 为了增加事件计数，重复执行算法
    for (int repeat = 0; repeat < 10; ++repeat) {
        if (algorithm_index == 1) {
            max_vertex = find_max_weight_vertex(graph);
        } else if (algorithm_index == 2) {
            max_vertex = find_max_rank_vertex(graph);
        }
    }

    retval = PAPI_stop(event_set, values);
    if (retval != PAPI_OK) {
        std::cerr << "PAPI stop counters error: " << PAPI_strerror(retval) << std::endl;
        exit(1);
    }

    // 清理事件集
    retval = PAPI_cleanup_eventset(event_set);
    if (retval != PAPI_OK) {
        std::cerr << "PAPI cleanup event set error: " << PAPI_strerror(retval) << std::endl;
        exit(1);
    }
    retval = PAPI_destroy_eventset(&event_set);
    if (retval != PAPI_OK) {
        std::cerr << "PAPI destroy event set error: " << PAPI_strerror(retval) << std::endl;
        exit(1);
    }
}

int main() {
    const char* filenames[N_TESTS] = {"synt", "road_graph", "stanford", "youtube", "syn_rmat"};

    // 初始化PAPI
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library initialization error!\n");
        exit(1);
    }
    // 存储各测试的算法性能结果
long long alg1_values[N_TESTS][7];
long long alg2_values[N_TESTS][7];

for (int n_test = 0; n_test < N_TESTS; n_test++) {
    CSR_graph graph;
    graph.read(filenames[n_test]);

    int max_vertex = -1;
    long long values[7] = {0, 0, 0, 0, 0, 0, 0};

    // 测试第一个算法
    monitor_performance(graph, max_vertex, 1, values);
    // 保存性能数据
    for (int i = 0; i < 7; ++i) {
        alg1_values[n_test][i] = values[i];
    }
    printf("Test: %s\n", filenames[n_test]);
    printf("Alg1 - Max weight vertex: %d\n", max_vertex);
    printf("Alg1 - PAPI_L1_DCM: %lld, PAPI_L1_ICM: %lld, PAPI_L1_TCM: %lld \nPAPI_L2_DCM: %lld, PAPI_L2_ICM: %lld, PAPI_L2_TCM: %lld \nPAPI_TOT_INS: %lld\n",
           alg1_values[n_test][0], alg1_values[n_test][1], alg1_values[n_test][2], alg1_values[n_test][3], alg1_values[n_test][4], alg1_values[n_test][5], alg1_values[n_test][6]);

    // 测试第二个算法
    monitor_performance(graph, max_vertex, 2, values);
    // 保存性能数据
    for (int i = 0; i < 7; ++i) {
        alg2_values[n_test][i] = values[i];
    }
    printf("Alg2 - Max rank vertex: %d\n", max_vertex);
    printf("Alg2 - PAPI_L1_DCM: %lld, PAPI_L1_ICM: %lld, PAPI_L1_TCM: %lld \nPAPI_L2_DCM: %lld, PAPI_L2_ICM: %lld, PAPI_L2_TCM: %lld \nPAPI_TOT_INS: %lld\n",
           alg2_values[n_test][0], alg2_values[n_test][1], alg2_values[n_test][2], alg2_values[n_test][3], alg2_values[n_test][4], alg2_values[n_test][5], alg2_values[n_test][6]);

    // 重置图结构，释放内存
    graph.reset();
}

// 关闭PAPI
PAPI_shutdown();

return 0;

}
