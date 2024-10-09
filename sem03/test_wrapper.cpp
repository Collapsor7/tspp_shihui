#include <iostream>
#include <vector>
#include <cstdio>
#include <papi.h>

#define N_TESTS 5

// 定义CSR图结构
class CSR_graph {
    int row_count;  // 图中顶点的数量
    unsigned int col_count;  // 图中边的数量

    std::vector<unsigned int> row_ptr;  // 行指针数组，存储每个顶点起始边的索引
    std::vector<int> col_ids;  // 列索引数组，存储每条边指向的目标顶点
    std::vector<double> vals;  // 边权重数组

public:
    // 从文件中读取CSR图数据
    void read(const char* filename) {
        FILE *graph_file = fopen(filename, "rb");
        fread(reinterpret_cast<char*>(&row_count), sizeof(int), 1, graph_file);
        fread(reinterpret_cast<char*>(&col_count), sizeof(unsigned int), 1, graph_file);

        std::cout << "Row_count = " << row_count << ", col_count = " << col_count << std::endl;

        row_ptr.resize(row_count + 1);
        col_ids.resize(col_count);
        vals.resize(col_count);

        fread(reinterpret_cast<char*>(row_ptr.data()), sizeof(unsigned int), row_count + 1, graph_file);
        fread(reinterpret_cast<char*>(col_ids.data()), sizeof(int), col_count, graph_file);
        fread(reinterpret_cast<char*>(vals.data()), sizeof(double), col_count, graph_file);
        fclose(graph_file);
    }

    // 打印某个顶点的所有边及其权重
    void print_vertex(int idx) {
        for (int col = row_ptr[idx]; col < row_ptr[idx + 1]; col++) {
            std::cout << col_ids[col] << " " << vals[col] << std::endl;
        }
        std::cout << std::endl;
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
        for (int col = row_ptr[vertex]; col < row_ptr[vertex + 1]; ++col) {
            edges.emplace_back(col_ids[col], vals[col]);
        }
        return edges;
    }
};

// 实现第一个算法：找到具有最大总权重（指向偶数顶点）的顶点
int find_max_weight_vertex(const CSR_graph &graph) {
    int max_vertex = -1;
    double max_weight = 0.0;

    for (int i = 0; i < graph.get_vertex_count(); ++i) {
        double total_weight = 0.0;
        for (const auto &edge : graph.get_edges(i)) {
            if (edge.first % 2 == 0) {  // 只计算指向偶数顶点的边
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
    double max_rank = 0.0;

    for (int i = 0; i < graph.get_vertex_count(); ++i) {
        double rank = 0.0;
        for (const auto &edge : graph.get_edges(i)) {
            int target_vertex = edge.first;
            double weight = edge.second;
            double vertex_weight = 0.0;

            // 直接计算目标顶点的权重 W(target_vertex)，不使用数组存储
            for (const auto &target_edge : graph.get_edges(target_vertex)) {
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

// 使用PAPI监控两个算法的性能
void monitor_performance(const CSR_graph &graph, int &max_weight_vertex, int &max_rank_vertex, long long *values) {
    // 初始化PAPI
    if (PAPI_library_init(PAPI_VER_CURRENT) != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library initialization error!\n");
        exit(1);
    }

    int event_set = PAPI_NULL;

    // 创建事件集并添加PAPI事件
    PAPI_create_eventset(&event_set);
    PAPI_add_event(event_set, PAPI_L1_TCM);
    PAPI_add_event(event_set, PAPI_L2_TCM);
    PAPI_add_event(event_set, PAPI_TOT_INS);  // 自选native事件

    // 第一个算法性能监测
    PAPI_start(event_set);
    max_weight_vertex = find_max_weight_vertex(graph);
    PAPI_stop(event_set, values);
    printf("Max weight vertex: %d\n", max_weight_vertex);
    printf("PAPI_L1_TCM: %lld, PAPI_L2_TCM: %lld, PAPI_TOT_INS: %lld\n", values[0], values[1], values[2]);

    // 第二个算法性能监测
    PAPI_start(event_set);
    max_rank_vertex = find_max_rank_vertex(graph);
    PAPI_stop(event_set, values);
    printf("Max rank vertex: %d\n", max_rank_vertex);
    printf("PAPI_L1_TCM: %lld, PAPI_L2_TCM: %lld, PAPI_TOT_INS: %lld\n", values[0], values[1], values[2]);

    // 修改后的清理操作，传递int类型，而非int*
    PAPI_cleanup_eventset(event_set);
    PAPI_destroy_eventset(&event_set);
    PAPI_shutdown();
}

int main() {
    const char* filenames[N_TESTS] = {"synt", "road_graph", "stanford", "youtube", "syn_rmat"};

    // 存储各测试的算法性能结果
    long long alg1_values[N_TESTS][3];
    long long alg2_values[N_TESTS][3];

    for (int n_test = 0; n_test < N_TESTS; n_test++) {
        CSR_graph graph;
        graph.read(filenames[n_test]);

        int max_weight_vertex = -1, max_rank_vertex = -1;
        long long values[3] = {0, 0, 0};

        // 测试并监控两个算法的性能
        monitor_performance(graph, max_weight_vertex, max_rank_vertex, values);

        // 保存性能数据
        for (int i = 0; i < 3; ++i) {
            alg1_values[n_test][i] = values[i];
        }

        // 再次执行第二个算法并保存性能数据
        monitor_performance(graph, max_weight_vertex, max_rank_vertex, values);
        for (int i = 0; i < 3; ++i) {
            alg2_values[n_test][i] = values[i];
        }

        // 输出结果
        printf("Test: %s\n", filenames[n_test]);
        printf("Alg1 L1 TCM: %lld, L2 TCM: %lld, TOT INS: %lld\n", alg1_values[n_test][0], alg1_values[n_test][1], alg1_values[n_test][2]);
        printf("Alg2 L1 TCM: %lld, L2 TCM: %lld, TOT INS: %lld\n", alg2_values[n_test][0], alg2_values[n_test][1], alg2_values[n_test][2]);

        // 重置图结构，释放内存
        graph.reset();
    }

    return 0;
}
