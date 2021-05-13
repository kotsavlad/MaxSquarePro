// Example of the application of OpenMP
// Search of (square_size x square_size)-window in the given matrix
// with the maximal sum of the absolute values of its elements.
// All functions returns the sought maximal sum.
// Functions with "parallel" suffix uses omp directives.

#include <iostream>
#include <fstream>
#include <algorithm>
#include <omp.h>
#include <deque>
#include <random>
#include <map>

using namespace std;

typedef double item_t;
//typedef int item_t;

// strait O(m*n*square_size^2) realization
item_t max_square_sum(item_t** matrix, int m, int n, int square_size) {
    item_t max_sum = -1;
    for (int i = 0; i <= m - square_size; i++) {
        for (int j = 0; j <= n - square_size; j++) {
            item_t square_sum = 0;
            for (int k = i; k < i + square_size; k++) {
                for (int l = j; l < j + square_size; l++) {
                    if (matrix[k][l] >= 0)
                        square_sum += matrix[k][l];
                    else
                        square_sum -= matrix[k][l];
                }
            }
            if (square_sum > max_sum)
                max_sum = square_sum;
        }
    }
    return max_sum;
}

// slightly improved version
item_t max_square_sum_b(item_t** matrix, int m, int n, int square_size) {
    item_t max_sum = -1;
    item_t* data_ptr;
    for (int i = 0; i <= m - square_size; i++) {
        for (int j = 0; j <= n - square_size; j++) {
            item_t square_sum = 0;
            for (int k = 0; k < square_size; k++) {
                data_ptr = &matrix[i + k][j];
                for (int l = 0; l < square_size; l++) {
                    if (*data_ptr >= 0)
                        square_sum += *data_ptr;
                    else
                        square_sum -= *data_ptr;
                    data_ptr++;
                }
            }
            if (square_sum > max_sum)
                max_sum = square_sum;
        }
    }
    return max_sum;
}

// improved O(m*n*square_size) algorithm using only last columns recalculation
item_t max_square_sum_opt1(item_t** matrix, int m, int n, int square_size) {
    item_t max_sum = -1;
    deque<item_t> sums;
    for (int i = 0; i <= m - square_size; i++) {
        sums.clear();
        item_t square_sum = 0;
        auto bottom_bound = i + square_size;
        for (int j = 0; j < square_size; j++) {
            item_t col_sum = 0;
            for (int k = i; k < bottom_bound; k++) {
                auto element = matrix[k][j];
                if (element > 0) {
                    col_sum += element;
                }
                else {
                    col_sum -= element;
                }
            }
            square_sum += col_sum;
            sums.push_back(col_sum);
        }
        if (square_sum > max_sum)
            max_sum = square_size;

        for (int j = square_size; j < n; j++) {
            item_t col_sum = 0;
            for (int k = i; k < bottom_bound; k++) {
                auto element = matrix[k][j];
                if (element > 0)
                    col_sum += element;
                else
                    col_sum -= element;
            }
            square_sum += col_sum - sums.front();
            if (square_sum > max_sum)
                max_sum = square_sum;
            sums.pop_front();
            sums.push_back(col_sum);
        }
    }
    return max_sum;
}

item_t max_square_sum_opt2(item_t** matrix, int m, int n, int square_size) {
    item_t max_sum = -1;
    auto* sums = new item_t[square_size];
    for (int i = 0; i <= m - square_size; i++) {
        //sums.clear();
        for (int k = 0; k < square_size; k++) {
            sums[k] = 0;
        }
        item_t square_sum = 0;
        auto bottom_bound = i + square_size;
        for (int j = 0; j < square_size; j++) {
            item_t col_sum = 0;
            for (int k = i; k < bottom_bound; k++) {
                auto element = matrix[k][j];
                if (element > 0)
                    col_sum += element;
                else
                    col_sum -= element;
            }
            square_sum += col_sum;
            sums[j] = col_sum;
        }
        if (square_sum > max_sum)
            max_sum = square_size;

        for (int j = square_size; j < n; j++) {
            item_t col_sum = 0;
            for (int k = i; k < bottom_bound; k++) {
                auto element = matrix[k][j];
                if (element > 0)
                    col_sum += element;
                else
                    col_sum -= element;
            }
            square_sum += col_sum - sums[j % square_size];
            if (square_sum > max_sum)
                max_sum = square_sum;
            sums[j % square_size] = col_sum;
        }
    }
    delete[] sums;
    return max_sum;
}

item_t max_square_sum_parallel_critical(item_t** matrix, int m, int n, int square_size, int thread_count = 4) {
    item_t max_sum = -1;
    item_t square_sum;
#pragma omp parallel for num_threads(thread_count) private(square_sum)
    for (int i = 0; i <= m - square_size; i++) {
        for (int j = 0; j <= n - square_size; j++) {
            square_sum = 0;
            for (int k = i; k < i + square_size; k++) {
                for (int l = j; l < j + square_size; l++) {
                    if (matrix[k][l] >= 0)
                        square_sum += matrix[k][l];
                    else
                        square_sum -= matrix[k][l];
                }
            }
            if (square_sum > max_sum)
#pragma omp critical
            {
                if (square_sum > max_sum)
                    max_sum = square_sum;
            }
        }
    }
    return max_sum;
}

item_t max_square_sum_parallel(item_t** matrix, int m, int n, int square_size, int thread_count = 4) {
    item_t square_sum;
    auto max_sums = new item_t[thread_count];
    item_t max_sum;
#pragma omp parallel num_threads(thread_count) private(square_sum, max_sum)
    {
        int id = omp_get_thread_num();
        max_sum = -1;
#pragma omp for
        for (int i = 0; i <= m - square_size; i++) {
            for (int j = 0; j <= n - square_size; j++) {
                square_sum = 0;
                for (int k = i; k < i + square_size; k++) {
                    for (int l = j; l < j + square_size; l++) {
                        if (matrix[k][l] >= 0)
                            square_sum += matrix[k][l];
                        else
                            square_sum -= matrix[k][l];
                    }
                }
                if (square_sum > max_sum)
                    max_sum = square_sum;
            }
        }
        max_sums[id] = max_sum;
    }
    max_sum = *std::max_element(max_sums, max_sums + thread_count);
    delete[] max_sums;
    return max_sum;
}

item_t max_square_sum_opt2_parallel(item_t** matrix, int m, int n, int square_size, int thread_count = 4) {
    auto* max_sums = new item_t[thread_count];
    item_t max_sum;
#pragma omp parallel num_threads(thread_count) private(max_sum)
    {
        int id = omp_get_thread_num();
        auto* sums = new item_t[square_size];
        max_sum = -1;
#pragma omp for
        for (int i = 0; i <= m - square_size; i++) {
            for (int k = 0; k < square_size; k++) {
                sums[k] = 0;
            }
            item_t square_sum = 0;
            auto bottom_bound = i + square_size;
            for (int j = 0; j < square_size; j++) {
                item_t col_sum = 0;
                for (int k = i; k < bottom_bound; k++) {
                    auto element = matrix[k][j];
                    col_sum += element > 0 ? element : -element;
                    //					col_sum += abs(element);
                }
                square_sum += col_sum;
                sums[j] = col_sum;
            }
            if (square_sum > max_sum)
                max_sum = square_size;

            for (int j = square_size; j < n; j++) {
                item_t col_sum = 0;
                for (int k = i; k < bottom_bound; k++) {
                    auto element = matrix[k][j];
                    if (element > 0)
                        col_sum += element;
                    else
                        col_sum -= element;
                }
                square_sum += col_sum - sums[j % square_size];
                if (square_sum > max_sum)
                    max_sum = square_sum;
                sums[j % square_size] = col_sum;
            }
        }
        max_sums[id] = max_sum;
        delete[] sums;
    }
    max_sum = *std::max_element(max_sums, max_sums + thread_count);
    delete[] max_sums;
    return max_sum;
}

// generates statistics and save it into specified file
void save_stats_to_file(item_t** matrix, int square_size, initializer_list<int> dims, initializer_list<int> num_threads, const string file_name) {
    fstream fs = fstream(file_name, ios::out);
    for (int m : dims) {
        auto start = omp_get_wtime();
        auto res = max_square_sum(matrix, m, m, square_size);
        auto duration = omp_get_wtime() - start;
        cout << "Dimension: " << m << endl;
        printf("Result in serial mode: %e, duration: %e\n", res, duration);

        for (int t : num_threads) {
            start = omp_get_wtime();
            res = max_square_sum_parallel(matrix, m, m, square_size);
            auto duration2 = omp_get_wtime() - start;
            auto speedup = duration / duration2;
            printf("Result in parallel mode for %d threads: %e, duration: %e, speedup: %.4f\n", t, res, duration2, speedup);
            fs << speedup << '\t';
        }
        fs << '\n';
    }
    fs.close();
}

int main() {
    const int MAX_DIM = 1000;
    int square_size = 30;
    auto matrix = new item_t * [MAX_DIM];
    default_random_engine rnd;
    for (int i = 0; i < MAX_DIM; i++) {
        matrix[i] = new item_t[MAX_DIM];
        for (int j = 0; j < MAX_DIM; j++)
            matrix[i][j] = (item_t)rnd();
    }

    printf("calculations started:\n");

    auto test_dim = MAX_DIM;
    auto start = omp_get_wtime();
    auto res = max_square_sum(matrix, test_dim, test_dim, square_size);
    auto duration = omp_get_wtime() - start;
    printf("max_square_sum result: %e, duration: %e\n", res, duration);

    start = omp_get_wtime();
    res = max_square_sum_b(matrix, test_dim, test_dim, square_size);
    auto duration2 = omp_get_wtime() - start;
    printf("max_square_sum_b result: %e, duration: %e, speedup: %.4f\n", res, duration2, duration / duration2);

    start = omp_get_wtime();
    res = max_square_sum_opt1(matrix, test_dim, test_dim, square_size);
    duration2 = omp_get_wtime() - start;
    printf("max_square_sum_opt1 result: %e, duration: %e, speedup: %.4f\n", res, duration2, duration / duration2);

    start = omp_get_wtime();
    res = max_square_sum_opt2(matrix, test_dim, test_dim, square_size);
    duration2 = omp_get_wtime() - start;
    printf("max_square_sum_opt2 result: %e, duration: %e, speedup: %.4f\n", res, duration2, duration / duration2);

    start = omp_get_wtime();
    res = max_square_sum_parallel_critical(matrix, test_dim, test_dim, square_size);
    duration2 = omp_get_wtime() - start;
    printf("max_square_sum_parallel_critical result: %e, duration: %e, speedup: %.4f\n", res, duration2, duration / duration2);

    start = omp_get_wtime();
    res = max_square_sum_parallel(matrix, test_dim, test_dim, square_size);
    duration2 = omp_get_wtime() - start;
    printf("max_square_sum_parallel result: %e, duration: %e, speedup: %.4f\n", res, duration2, duration / duration2);

    start = omp_get_wtime();
    res = max_square_sum_opt2_parallel(matrix, test_dim, test_dim, square_size);
    duration2 = omp_get_wtime() - start;
    printf("max_square_sum_opt2_parallel result: %e, duration: %e, speedup: %.4f\n", res, duration2,
        duration / duration2);

    //    auto dims = { 100, 200, 300, 400, 500, MAX_DIM };
    //	auto num_threads = {2, 4, 6, 8, 10, 20 };
    //    auto file_name = "d:/result.txt";
    //    save_stats_to_file(matrix, square_size, dims, num_threads, file_name);

        // Memory cleanup!
    for (size_t i = 0; i < MAX_DIM; i++)
        delete[] matrix[i];
    delete[] matrix;
    printf("calculations finished:\n");
}
