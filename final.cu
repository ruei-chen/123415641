#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <chrono>

using namespace std;
using namespace std::chrono;

// 定義 Point 類別
class Point
{
public:
    int pointId;
    int clusterId;
    int dimensions;
    double *values;

    // 預設建構函數
    Point(int id, double *vals, int dim) : pointId(id), clusterId(-1), dimensions(dim)
    {
        values = new double[dim];
        memcpy(values, vals, dim * sizeof(double));
    }

    // 深拷貝建構函數
    Point(const Point &other) : pointId(other.pointId), clusterId(other.clusterId), dimensions(other.dimensions)
    {
        values = new double[dimensions];
        memcpy(values, other.values, dimensions * sizeof(double));
    }

    // 深拷貝賦值運算符
    Point &operator=(const Point &other)
    {
        if (this == &other)
            return *this; // 避免自我賦值
        delete[] values;  // 釋放原來的記憶體
        pointId = other.pointId;
        clusterId = other.clusterId;
        dimensions = other.dimensions;
        values = new double[dimensions];
        memcpy(values, other.values, dimensions * sizeof(double));
        return *this;
    }

    // 解構函數
    ~Point()
    {
        delete[] values;
    }
};

// 自定義 atomicAddDouble，用於支援 double 原子加法
__device__ double atomicAddDouble(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);

    return __longlong_as_double(old);
}

// CUDA 核心函數：計算最近叢集
// points 每個點的座標  centroid 每個群的中心  clusterassignment 每個點哪一群   total_points 點的總數
__global__ void assignClusters(double *points, double *centroids, int *clusterAssignments,
                               int total_points, int K, int dimensions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_points)
        return;

    int nearestClusterId = -1;
    double minDist = 10000;

    for (int k = 0; k < K; ++k)
    {
        double sum = 0.0;
        for (int d = 0; d < dimensions; ++d)
        {
            double diff = points[idx * dimensions + d] - centroids[k * dimensions + d];
            sum += diff * diff;
        }
        double distance = sqrt(sum);
        if (distance < minDist)
        {
            minDist = distance;
            nearestClusterId = k;
        }
    }
    clusterAssignments[idx] = nearestClusterId;
}

// CUDA 核心函數：計算新的叢集中心
// points 每個點的座標  centroid 每個群的中心  clusterassignment 每個點哪一群   total_points 點的總數
__global__ void computeNewCentroids(double *points, double *centroids, int *clusterAssignments,
                                    int *clusterSizes, int total_points, int K, int dimensions)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_points)
        return;

    int clusterId = clusterAssignments[idx];
    for (int d = 0; d < dimensions; ++d)
    {
        atomicAddDouble(&centroids[clusterId * dimensions + d], points[idx * dimensions + d]);
    }
    atomicAdd(&clusterSizes[clusterId], 1);
}

// 定義 KMeans 類別
class KMeans
{
private:
    int K, dimensions, total_points;
    string output_dir;

public:
    KMeans(int K, string output_dir) : K(K), output_dir(output_dir) {}

    void run(vector<Point> &all_points)
    {
        total_points = all_points.size();
        dimensions = all_points[0].dimensions;

        // 主機記憶體分配
        double *host_points = new double[total_points * dimensions];
        double *host_centroids = new double[K * dimensions];
        int *host_clusterAssignments = new int[total_points];
        int *host_clusterSizes = new int[K]();

        // 初始化點座標與叢集中心
        for (int i = 0; i < total_points; ++i)
        {
            memcpy(&host_points[i * dimensions], all_points[i].values, dimensions * sizeof(double));
        }

        for (int i = 0; i < K; ++i)
        {
            int randomPointIndex = rand() % total_points;
            memcpy(&host_centroids[i * dimensions], all_points[randomPointIndex].values, dimensions * sizeof(double));
        }

        // 設備記憶體分配
        double *device_points, *device_centroids;
        int *device_clusterAssignments, *device_clusterSizes;
        cudaMalloc(&device_points, total_points * dimensions * sizeof(double));
        cudaMalloc(&device_centroids, K * dimensions * sizeof(double));
        cudaMalloc(&device_clusterAssignments, total_points * sizeof(int));
        cudaMalloc(&device_clusterSizes, K * sizeof(int));

        // 主機資料拷貝到設備
        cudaMemcpy(device_points, host_points, total_points * dimensions * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(device_centroids, host_centroids, K * dimensions * sizeof(double), cudaMemcpyHostToDevice);

        const int threadsPerBlock = 256;
        const int blocksPerGrid = (total_points + threadsPerBlock - 1) / threadsPerBlock;

        for (int iter = 0; iter < 100; ++iter)
        {
            // 清空叢集大小
            // cout << "Iter - " << iter + 1 << "/ 100" << endl;
            cudaMemset(device_clusterSizes, 0, K * sizeof(int));
            cudaMemset(device_clusterAssignments, -1, total_points * sizeof(int));

            // 分配最近叢集
            assignClusters<<<blocksPerGrid, threadsPerBlock>>>(device_points, device_centroids, device_clusterAssignments,
                                                               total_points, K, dimensions);
            cudaDeviceSynchronize();

            // 計算新叢集中心
            // points 每個點的座標   clusterassignment 每個點哪一群   total_points 點的總數
            cudaMemset(device_centroids, 0, K * dimensions * sizeof(double));
            computeNewCentroids<<<blocksPerGrid, threadsPerBlock>>>(device_points, device_centroids, device_clusterAssignments,
                                                                    device_clusterSizes, total_points, K, dimensions);
            cudaDeviceSynchronize();

            // 更新叢集中心平均值
            cudaMemcpy(host_centroids, device_centroids, K * dimensions * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_clusterSizes, device_clusterSizes, K * sizeof(int), cudaMemcpyDeviceToHost);

            for (int k = 0; k < K; ++k)
            {
                for (int d = 0; d < dimensions; ++d)
                {
                    if (host_clusterSizes[k] > 0)
                    {
                        // centroid 每個群的中心  clustersize 每個群的點數
                        host_centroids[k * dimensions + d] /= host_clusterSizes[k];
                    }
                }
            }

            // 拷貝更新後的叢集中心到設備
            cudaMemcpy(device_centroids, host_centroids, K * dimensions * sizeof(double), cudaMemcpyHostToDevice);
        }

        // 複製結果到主機
        cudaMemcpy(host_clusterAssignments, device_clusterAssignments, total_points * sizeof(int), cudaMemcpyDeviceToHost);

        // 儲存結果
        ofstream centroidsFile(output_dir + "/centroids.txt");
        ofstream pointsFile(output_dir + "/points.txt");
        for (int k = 0; k < K; ++k)
        {
            for (int d = 0; d < dimensions; ++d)
            {
                centroidsFile << host_centroids[k * dimensions + d];
                if (d < dimensions - 1)
                {
                    centroidsFile << " "; // 座標之間用空格分隔
                }
            }
            centroidsFile << endl; // 每個重心座標換行
        }
        centroidsFile.close();

        // 儲存點所屬叢集編號
        for (int i = 0; i < total_points; ++i)
        {
            pointsFile << (host_clusterAssignments[i] + 1) << endl; // 群編號調整為 1-K
        }
        pointsFile.close();
        // 釋放設備記憶體
        cudaFree(device_points);
        cudaFree(device_centroids);
        cudaFree(device_clusterAssignments);
        cudaFree(device_clusterSizes);

        // 釋放主機記憶體
        delete[] host_points;
        delete[] host_centroids;
        delete[] host_clusterAssignments;
        delete[] host_clusterSizes;
    }
};

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cout << "Usage: ./final <INPUT> <K> <OUTPUT_DIR>" << endl;
        return 1;
    }

    string input_file = argv[1];
    int K = atoi(argv[2]);
    string output_dir = argv[3];

    ifstream infile(input_file);
    if (!infile.is_open())
    {
        cout << "Error: Unable to open file " << input_file << endl;
        return 1;
    }

    vector<Point> all_points;
    string line;
    int pointId = 0;

    // 跳過表頭
    getline(infile, line);

    while (getline(infile, line))
    {
        stringstream ss(line);
        string cell;
        vector<double> values;

        int columnIdx = 0;
        while (getline(ss, cell, '\t'))
        { // 使用 '\t' 作為分隔符
            columnIdx++;

            // 選擇只處理數值列，例如第 4 到第 10 列
            if (columnIdx >= 5 && columnIdx <= 20 && columnIdx != 8)
            {
                try
                {
                    values.push_back(stod(cell));
                }
                catch (const invalid_argument &e)
                {
                    // 忽略非數值字段
                    continue;
                }
            }
        }

        if (!values.empty())
        {
            all_points.emplace_back(pointId++, values.data(), values.size());
        }
    }

    infile.close();

    if (all_points.empty())
    {
        cerr << "Error: No valid points parsed from the input file. Please check the file format and column indices." << endl;
        return 1;
    }

    cout << "\nData fetched successfully!" << endl
         << endl;

    auto start_time = high_resolution_clock::now();

    // 執行 KMeans
    for (int i = 3; i <= K; i++)
    {
        KMeans kmeans(K, output_dir);
        kmeans.run(all_points);
    }

    auto end_time = high_resolution_clock::now(); // End timing for iteration
    auto iter_duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Program completed in " << iter_duration.count() << " ms" << endl;

    return 0;
}
