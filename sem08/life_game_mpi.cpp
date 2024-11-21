#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>

#define ALIVE 1
#define DEAD 0

const int GRID_HEIGHT = 32;
const int GRID_WIDTH = 32;

const int K = 10;
const int MAX_ITER = 100;

#define INDEX(i, j) ((i) * GRID_WIDTH + (j))

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double start_time = MPI_Wtime();

    int base_height = GRID_HEIGHT / size;
    int extra = GRID_HEIGHT % size;
    int start_row, end_row;
    int local_height;

    if (rank < extra) {
        local_height = base_height + 1;
        start_row = rank * local_height;
    } else {
        local_height = base_height;
        start_row = rank * local_height + extra;
    }
    end_row = start_row + local_height - 1;

    std::vector<int> current((local_height + 2) * GRID_WIDTH, DEAD);
    std::vector<int> next((local_height + 2) * GRID_WIDTH, DEAD);

    std::vector<int> global_grid;
    if (rank == 0) {
        global_grid.resize(GRID_HEIGHT * GRID_WIDTH, DEAD);

        int center_i = GRID_HEIGHT / 2;
        int center_j = GRID_WIDTH / 2;

        int glider[5][2] = {
            {center_i, center_j + 1},
            {center_i + 1, center_j + 2},
            {center_i + 2, center_j},
            {center_i + 2, center_j + 1},
            {center_i + 2, center_j + 2}
        };
        for (auto& cell : glider) {
            int x = cell[0];
            int y = cell[1];
            if (x >= 0 && x < GRID_HEIGHT && y >= 0 && y < GRID_WIDTH) {
                global_grid[INDEX(x, y)] = ALIVE;
            }
        }

        std::cout << "Initial Grid State:" << std::endl;
        for (int i = 0; i < GRID_HEIGHT; ++i) {
            for (int j = 0; j < GRID_WIDTH; ++j) {
                std::cout << (global_grid[INDEX(i, j)] == ALIVE ? 'O' : '.') << " ";
            }
            std::cout << std::endl;
        }
    }

    std::vector<int> send_counts(size);
    std::vector<int> send_displs(size);
    int offset = 0;

    for (int p = 0; p < size; ++p) {
        int p_local_height = (p < extra) ? base_height + 1 : base_height;
        send_counts[p] = p_local_height * GRID_WIDTH;
        send_displs[p] = offset;
        offset += send_counts[p];
    }

    MPI_Scatterv(rank == 0 ? global_grid.data() : nullptr, send_counts.data(), send_displs.data(), MPI_INT,
                 &current[GRID_WIDTH], local_height * GRID_WIDTH, MPI_INT, 0, comm);

    std::vector<int> prev_current = current;
    std::vector<int> prev_global_grid_flat;
    if (rank == 0) {
        prev_global_grid_flat.resize(GRID_HEIGHT * GRID_WIDTH, DEAD);
    }

    int iter = 0;
    int stable = 0;

    while (iter < MAX_ITER) {
        ++iter;

        MPI_Request requests[4];

        int up_rank = rank - 1;
        int down_rank = rank + 1;

        int has_up = (up_rank >= 0) ? 1 : 0;
        int has_down = (down_rank < size) ? 1 : 0;

        const int TAG_SEND_UP = 0;
        const int TAG_SEND_DOWN = 1;

        int req_count = 0;

        if (has_up) {
            MPI_Isend(&current[1 * GRID_WIDTH], GRID_WIDTH, MPI_INT, up_rank, TAG_SEND_DOWN, comm, &requests[req_count++]);
            MPI_Irecv(&current[0 * GRID_WIDTH], GRID_WIDTH, MPI_INT, up_rank, TAG_SEND_UP, comm, &requests[req_count++]);
        }

        if (has_down) {
            MPI_Isend(&current[local_height * GRID_WIDTH], GRID_WIDTH, MPI_INT, down_rank, TAG_SEND_UP, comm, &requests[req_count++]);
            MPI_Irecv(&current[(local_height + 1) * GRID_WIDTH], GRID_WIDTH, MPI_INT, down_rank, TAG_SEND_DOWN, comm, &requests[req_count++]);
        }

        if (req_count > 0) {
            MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);
        }

        for (int i = 1; i <= local_height; ++i) {
            for (int j = 0; j < GRID_WIDTH; ++j) {
                int neighbors = 0;
                for (int di = -1; di <= 1; ++di) {
                    int ni = i + di;
                    if (ni < 1 || ni > local_height) {
                        if ((di == -1 && !has_up) || (di == 1 && !has_down)) {
                            continue;
                        }
                    }
                    for (int dj = -1; dj <= 1; ++dj) {
                        if (di == 0 && dj == 0) continue;
                        int nj = j + dj;
                        if (nj < 0 || nj >= GRID_WIDTH) {
                            continue;
                        }
                        neighbors += current[INDEX(ni, nj)];
                    }
                }
                int idx = INDEX(i, j);
                if (current[idx] == ALIVE) {
                    next[idx] = (neighbors == 2 || neighbors == 3) ? ALIVE : DEAD;
                } else {
                    next[idx] = (neighbors == 3) ? ALIVE : DEAD;
                }
            }
        }

        current.swap(next);

        int local_alive = 0;
        for (int i = 1; i <= local_height; ++i) {
            for (int j = 0; j < GRID_WIDTH; ++j) {
                if (current[INDEX(i, j)] == ALIVE) {
                    ++local_alive;
                }
            }
        }

        int total_alive = 0;
        MPI_Reduce(&local_alive, &total_alive, 1, MPI_INT, MPI_SUM, 0, comm);

        int local_grid_size = local_height * GRID_WIDTH;
        std::vector<int> local_grid_flat(local_grid_size);
        for (int i = 1; i <= local_height; ++i) {
            for (int j = 0; j < GRID_WIDTH; ++j) {
                local_grid_flat[(i - 1) * GRID_WIDTH + j] = current[INDEX(i, j)];
            }
        }

        std::vector<int> recv_counts(size);
        std::vector<int> recv_displs(size);

        int offset_recv = 0;
        for (int p = 0; p < size; ++p) {
            int p_local_height = (p < extra) ? base_height + 1 : base_height;
            recv_counts[p] = p_local_height * GRID_WIDTH;
            recv_displs[p] = offset_recv;
            offset_recv += recv_counts[p];
        }

        std::vector<int> global_grid_flat;
        if (rank == 0) {
            global_grid_flat.resize(GRID_HEIGHT * GRID_WIDTH);
        }

        MPI_Gatherv(local_grid_flat.data(), local_grid_size, MPI_INT,
                    global_grid_flat.data(), recv_counts.data(), recv_displs.data(), MPI_INT, 0, comm);

        int global_grid_stable = 0;
        if (rank == 0) {
            global_grid_stable = (global_grid_flat == prev_global_grid_flat) ? 1 : 0;
            prev_global_grid_flat = global_grid_flat;
        }

        MPI_Bcast(&global_grid_stable, 1, MPI_INT, 0, comm);

        if (global_grid_stable == 1 && iter >= K) {
            break;
        }

        prev_current = current;

        if (rank == 0) {
            std::cout << "\nGrid State at Iteration " << iter << ":\n";
            for (int i = 0; i < GRID_HEIGHT; ++i) {
                for (int j = 0; j < GRID_WIDTH; ++j) {
                    int idx = i * GRID_WIDTH + j;
                    std::cout << (global_grid_flat[idx] == ALIVE ? 'O' : '.') << " ";
                }
                std::cout << std::endl;
            }
            std::cout << "Total alive cells: " << total_alive << std::endl;
        }

    }

    double end_time = MPI_Wtime();
    if (rank == 0) {
        std::cout << "Total iterations times: " << iter << std::endl;
        double total_time = end_time - start_time;
        std::cout << "Total execution time: " << total_time << " seconds" << std::endl;
    }

    MPI_Finalize();
    return 0;
}
