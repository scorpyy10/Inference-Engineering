
#include <iostream>
#include <vector>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <algorithm>

using namespace std;
using namespace chrono;

const int N = 8192;

int main() {
    vector<int> matrix(N * N);

    // Initialize with non-constant data
    for (size_t i = 0; i < matrix.size(); i++) {
        matrix[i] = i % 100;
    }

    uint64_t row_sum = 0;
    uint64_t col_sum = 0;

    // Warm-up
    for (size_t i = 0; i < matrix.size(); i++) {
        row_sum += matrix[i];
    }
    row_sum = 0;
    // Row-major
    auto start = steady_clock::now();

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            row_sum += matrix[i * N + j];
        }
    }

    auto end = steady_clock::now();

    double row_time =
        duration<double, milli>(end - start).count();

    // Column-major
    start = steady_clock::now();

    for (int j = 0; j < N; j++) {
        for (int i = 0; i < N; i++) {
            col_sum += matrix[i * N + j];
        }
    }

    end = steady_clock::now();

    double col_time =
        duration<double, milli>(end - start).count();

    cout << fixed << setprecision(2);

    cout << "Row-major: " << row_time << " ms\n";
    cout << "Column-major: " << col_time << " ms\n";

    cout << "Slowdown: "
         << col_time / row_time << "x\n";

    cout << "Checksums: "
         << row_sum << ", " << col_sum << "\n";
}
