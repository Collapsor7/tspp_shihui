#include <iostream> 
#include <vector> 
#include <papi.h> 
#include <stdio.h> 
#include <stdlib.h> 

class CSR_graph { 
int row_count; // Number of vertices in graph 
unsigned int col_count; // Number of edges in graph 

std::vector<unsigned int> row_ptr; // Row pointer 
std::vector<int> col_ids; // Column indices 
std::vector<double> vals; // Values of the edges 

public: 
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

double calculate_weight_sum(int vertex) { 
double weight_sum = 0.0; 
for (unsigned int col = row_ptr[vertex]; col < row_ptr[vertex + 1]; col++) { 
if (col_ids[col] % 2 == 0) { // Check if the connected vertex is even 
weight_sum += vals[col]; // Sum the weights of edges connected to even vertices 
} 
} 
return weight_sum; 
} 

void calculate_ranks() { 
std::vector<double> ranks(row_count, 0.0); // Ranks initialized to 0 

for (int vertex = 0; vertex < row_count; ++vertex) { 
for (unsigned int col = row_ptr[vertex]; col < row_ptr[vertex + 1]; col++) { 
ranks[vertex] += vals[col]; // Calculate rank as sum of connected edge weights 
} 
} 

// Output ranks for each vertex 
for (int i = 0; i < row_count; ++i) { 
std::cout << "Rank of vertex " << i << ": " << ranks[i] << std::endl; 
} 
} 

int get_vertex_with_max_weight() { 
double max_weight = -1.0; 
int max_vertex = -1; 

for (int vertex = 0; vertex < row_count; ++vertex) { 
double weight_sum = calculate_weight_sum(vertex); 
if (weight_sum > max_weight) { 
max_weight = weight_sum; 
max_vertex = vertex; 
} 
} 

return max_vertex; 
} 
}; 

#define N_TESTS 5 

void handle_error(int retval, const char *str) { 
if (retval != PAPI_OK) { 
fprintf(stderr, "%s: PAPI error %d: %s\n", str, retval, PAPI_strerror(retval)); 
exit(1); 
} 
} 

int main() { 
const char* filenames[N_TESTS] = { 
"synt", // 示例文件名 
"road_graph", // 示例文件名 
"stanford", // 示例文件名 
"youtube", // 示例文件名 
"syn_rmat" // 示例文件名 
}; 

// 初始化 PAPI 
int event_set = PAPI_NULL; 
long long values[2]; 

handle_error(PAPI_library_init(PAPI_VER_CURRENT), "PAPI library init"); 
handle_error(PAPI_create_eventset(&event_set), "create eventset"); 
handle_error(PAPI_add_event(event_set, PAPI_L1_DCM), "add PAPI_L1_DCM"); 
//handle_error(PAPI_add_event(event_set, PAPI_L2_DCM), "add PAPI_L2_DCM"); 

// 遍历测试文件 
for (int n_test = 0; n_test < N_TESTS; n_test++) { 
CSR_graph graph; 
graph.read(filenames[n_test]); 

handle_error(PAPI_start(event_set), "PAPI start"); 

// 找到与偶数编号顶点相连的最大权重顶点 
int vertex = graph.get_vertex_with_max_weight(); 
std::cout << "Vertex with maximum weight sum: " << vertex << std::endl; 

// 计算所有顶点的等级 
graph.calculate_ranks(); 

handle_error(PAPI_stop(event_set, values), "PAPI stop"); 
std::cout << "PAPI_L1_DCM = " << values[0] << std::endl; 
//std::cout << "PAPI_L2_DCM = " << values[1] << std::endl; 
} 

// 清理 PAPI 事件集 
handle_error(PAPI_cleanup_eventset(event_set), "cleanup eventSet"); 
handle_error(PAPI_destroy_eventset(&event_set), "destroy eventset"); 

return 0; 
} 
