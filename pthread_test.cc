#include <pthread.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <chrono>
#include <cfloat>
#include <atomic>

using namespace std;
using namespace std::chrono;

class Point;
class Cluster;
class KMeans;

// Thread data structure
struct ThreadData {
    KMeans* kmeans;
    int thread_id;
    pthread_mutex_t work_mutex;
    pthread_cond_t work_cond;
    pthread_cond_t completion_cond;
    bool has_work;

    ThreadData(KMeans* k, int id) : kmeans(k), thread_id(id), has_work(false) {
        pthread_mutex_init(&work_mutex, NULL);
        pthread_cond_init(&work_cond, NULL);
        pthread_cond_init(&completion_cond, NULL);
    }

    ~ThreadData() {
        pthread_mutex_destroy(&work_mutex);
        pthread_cond_destroy(&work_cond);
        pthread_cond_destroy(&completion_cond);
    }
};
struct CentroidUpdateData {
    vector<Cluster>* clusters;
    int cluster_start;
    int cluster_end;
    int dimensions;
};
class Point {
private:
    int pointId, clusterId;
    int dimensions;
    vector<double> values;

    vector<double> lineToVec(string &line) {
        vector<double> values;
        string tmp = "";

        for (int i = 0; i < (int)line.length(); i++) {
            if ((48 <= int(line[i]) && int(line[i]) <= 57) || 
                line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e') {
                tmp += line[i];
            }
            else if (tmp.length() > 0) {
                values.push_back(stod(tmp));
                tmp = "";
            }
        }
        if (tmp.length() > 0) {
            values.push_back(stod(tmp));
        }
        return values;
    }

public:
    Point(int id, string line) {
        pointId = id;
        values = lineToVec(line);
        dimensions = values.size();
        clusterId = 0;
    }

    int getDimensions() { return dimensions; }
    int getCluster() { return clusterId; }
    int getID() { return pointId; }
    void setCluster(int val) { clusterId = val; }
    double getVal(int pos) { return values[pos]; }
};

class Cluster {
private:
    int clusterId;
    vector<double> centroid;
    vector<Point*> points;

public:
    Cluster(int clusterId, Point& centroid) {
        this->clusterId = clusterId;
        for (int i = 0; i < centroid.getDimensions(); i++) {
            this->centroid.push_back(centroid.getVal(i));
        }
        this->addPoint(&centroid);
    }

    void addPoint(Point* p) {
        p->setCluster(this->clusterId);
        points.push_back(p);
    }
    bool removePoint(int pointId)
    {
        int size = points.size();

        for (int i = 0; i < size; i++)
        {
            if (points[i]->getID() == pointId)
            {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }

    void removeAllPoints() { points.clear(); }

    int getId() { return clusterId; }
    Point* getPoint(int pos) { return points[pos]; }
    int getSize() { return points.size(); }
    double getCentroidByPos(int pos) { return centroid[pos]; }
    void setCentroidByPos(int pos, double val) { centroid[pos] = val; }
};

class KMeans {
private:
    int K, iters, dimensions, total_points;
    vector<Cluster> clusters;
    string output_dir;
    int num_threads;
    atomic<bool> done;
    atomic<bool> should_terminate;
    vector<ThreadData*> thread_data;
    vector<pthread_t> threads;

    struct BatchData {
        vector<Point*> points;
        int start_idx;
        int end_idx;
    };
    vector<BatchData> batches;

    static void* thread_function(void* arg) {
        ThreadData* data = (ThreadData*)arg;
        KMeans* kmeans = data->kmeans;
        
        while (!kmeans->should_terminate) {
            pthread_mutex_lock(&data->work_mutex);
            while (!data->has_work && !kmeans->should_terminate) {
                pthread_cond_wait(&data->work_cond, &data->work_mutex);
            }
            if (kmeans->should_terminate) {
                pthread_mutex_unlock(&data->work_mutex);
                break;
            }
            pthread_mutex_unlock(&data->work_mutex);

            bool local_done = true;
            BatchData& batch = kmeans->batches[data->thread_id];
            
            for (Point* point : batch.points) {
                int current_cluster = point->getCluster();
                int nearest_cluster = kmeans->getNearestClusterId(*point);
                
                if (current_cluster != nearest_cluster) {
                    point->setCluster(nearest_cluster);
                    local_done = false;
                }
            }

            if (!local_done) {
                kmeans->done.store(false, memory_order_relaxed);
            }

            pthread_mutex_lock(&data->work_mutex);
            data->has_work = false;
            pthread_cond_signal(&data->completion_cond);
            pthread_mutex_unlock(&data->work_mutex);
        }
        return NULL;
    }

    static void* updateCentroid(void* arg) {
        CentroidUpdateData* data = (CentroidUpdateData*)arg;
        
        for(int i = data->cluster_start; i < data->cluster_end; i++) {
            Cluster& cluster = (*(data->clusters))[i];
            int clusterSize = cluster.getSize();
            if (clusterSize > 0) {
                vector<double> newCentroid(data->dimensions, 0.0);
                for (int j = 0; j < clusterSize; j++) {
                    for (int d = 0; d < data->dimensions; d++) {
                        newCentroid[d] += cluster.getPoint(j)->getVal(d);
                    }
                }
                for (int d = 0; d < data->dimensions; d++) {
                    cluster.setCentroidByPos(d, newCentroid[d] / clusterSize);
                }
            }
        }
        return NULL;
    }

    int getNearestClusterId(Point& point) {
        double min_dist = DBL_MAX;
        int nearest_cluster = -1;

        for (int i = 0; i < K; i++) {
            double dist = 0.0;
            for (int j = 0; j < dimensions; j++) {
                double diff;
                diff=clusters[i].getCentroidByPos(j)-point.getVal(j);
                dist+=diff*diff;
                // dist+=pow(clusters[i].getCentroidByPos(j)-point.getVal(j),2);
            }
            dist = sqrt(dist);

            if (dist < min_dist) {
                min_dist = dist;
                nearest_cluster = clusters[i].getId();
            }
        }
        return nearest_cluster;
    }

    void clearClusters() {
        for (int i = 0; i < K; i++) {
            clusters[i].removeAllPoints();
        }
    }

public:
    KMeans(int K, int iterations, string output_dir, int num_threads =4) 
        : K(K), iters(iterations), output_dir(output_dir), 
          num_threads(num_threads), done(false), should_terminate(false) {
        
        threads.resize(num_threads);
        thread_data.resize(num_threads);
        
        for (int i = 0; i < num_threads; i++) {
            thread_data[i] = new ThreadData(this, i);
            pthread_create(&threads[i], NULL, thread_function, thread_data[i]);
        }
    }

    ~KMeans() {
        should_terminate.store(true);
        
        for (int i = 0; i < num_threads; i++) {
            pthread_mutex_lock(&thread_data[i]->work_mutex);
            thread_data[i]->has_work = true;
            pthread_cond_signal(&thread_data[i]->work_cond);
            pthread_mutex_unlock(&thread_data[i]->work_mutex);
            
            pthread_join(threads[i], NULL);
            delete thread_data[i];
        }
    }

    void run(vector<Point>& all_points) {
        total_points = all_points.size();
        dimensions = all_points[0].getDimensions();

        // Initializing Clusters
        vector<int> used_pointIds;
        for (int i = 1; i <= K; i++) {
            while (true) {
                int index = rand() % total_points;
                if (find(used_pointIds.begin(), used_pointIds.end(), index) == used_pointIds.end()) {
                    used_pointIds.push_back(index);
                    all_points[index].setCluster(i);
                    Cluster cluster(i, all_points[index]);
                    clusters.push_back(cluster);
                    break;
                }
            }
        }
        cout << "Clusters initialized = " << clusters.size() << endl << endl;
        cout << "Running K-Means Clustering.." << endl;

        // Prepare batches
        int points_per_thread = total_points / num_threads;
        batches.resize(num_threads);
        
        for (int i = 0; i < num_threads; i++) {
            int start = i * points_per_thread;
            int end = (i == num_threads - 1) ? total_points : (i + 1) * points_per_thread;
            
            batches[i].start_idx = start;
            batches[i].end_idx = end;
            for (int j = start; j < end; j++) {
                batches[i].points.push_back(&all_points[j]);
            }
        }

        int iter = 1;
        while (true) {
            cout << "Iter - " << iter << "/" << iters << endl;
            done.store(true);

            // Assign work to threads
            for (int i = 0; i < num_threads; i++) {
                pthread_mutex_lock(&thread_data[i]->work_mutex);
                thread_data[i]->has_work = true;
                pthread_cond_signal(&thread_data[i]->work_cond);
                pthread_mutex_unlock(&thread_data[i]->work_mutex);
            }

            // Wait for all threads to complete
            for (int i = 0; i < num_threads; i++) {
                pthread_mutex_lock(&thread_data[i]->work_mutex);
                while (thread_data[i]->has_work) {
                    pthread_cond_wait(&thread_data[i]->completion_cond, 
                                    &thread_data[i]->work_mutex);
                }
                pthread_mutex_unlock(&thread_data[i]->work_mutex);
            }

            clearClusters();
            for (Point& point : all_points) {
                clusters[point.getCluster() - 1].addPoint(&point);
            }

            // Update centroids
            vector<pthread_t> centroid_threads(num_threads);
            vector<CentroidUpdateData> centroid_data(num_threads);
            int clusters_per_thread = (K + num_threads - 1) / num_threads;

            for (int i = 0; i < num_threads; i++) {
                centroid_data[i].clusters = &clusters;
                centroid_data[i].cluster_start = i * clusters_per_thread;
                centroid_data[i].cluster_end = min((i + 1) * clusters_per_thread, K);
                centroid_data[i].dimensions = dimensions;
                
                pthread_create(&centroid_threads[i], NULL, updateCentroid, &centroid_data[i]);
            }

            for (int i = 0; i < num_threads; i++) {
                pthread_join(centroid_threads[i], NULL);
            }

            if (done.load() || iter >= iters) {
                cout << "Clustering completed in iteration : " << iter << endl << endl;
                break;
            }
            iter++;
        }

        // Write results to files
        ofstream pointsFile;
        pointsFile.open(output_dir + "/" + to_string(K) + "-points.txt", ios::out);
        for (int i = 0; i < total_points; i++) {
            pointsFile << all_points[i].getCluster() << endl;
        }
        pointsFile.close();

        ofstream outfile;
        outfile.open(output_dir + "/" + to_string(K) + "-clusters.txt");
        if (outfile.is_open()) {
            for (int i = 0; i < K; i++) {
                cout << "Cluster " << clusters[i].getId() << " centroid : ";
                for (int j = 0; j < dimensions; j++) {
                    cout << clusters[i].getCentroidByPos(j) << " ";
                    outfile << clusters[i].getCentroidByPos(j) << " ";
                }
                cout << endl;
                outfile << endl;
            }

            // Calculate total distance
            double sum = 0.0;
            for (int i = 0; i < total_points; i++) {
                int type = all_points[i].getCluster();
                double distance = 0.0;
                for (int j = 0; j < dimensions; j++) {
                    double cluster_value = clusters[type - 1].getCentroidByPos(j);
                    double point_value = all_points[i].getVal(j);
                    double diff=cluster_value - point_value;
                    distance+=diff*diff;
                    // distance += pow(cluster_value - point_value, 2);
                }
                sum += sqrt(distance);
            }
            cout << sum << endl;
            outfile << "distance : " << sum;
            outfile.close();
        }
    }
};

int main(int argc, char **argv) {
    auto start_time = high_resolution_clock::now();

    if (argc != 5) {
        cout << "Error: command-line argument count mismatch. \n ./kmeans <INPUT> <K> <OUT-DIR>" << endl;
        return 1;
    }

    string filename = argv[1];
    int K = atoi(argv[2]);
    string output_dir = argv[3];
    int thread_num=atoi(argv[4]);

    ifstream infile(filename.c_str());
    if (!infile.is_open()) {
        cout << "Error: Failed to open file." << endl;
        return 1;
    }

    vector<Point> all_points;
    string line;
    int pointId = 1;
    bool isFirstLine = true;

    while (getline(infile, line)) {
        if (isFirstLine) {
            isFirstLine = false;
            continue;
        }

        stringstream ss(line);
        string cell;
        vector<string> row;

        while (getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        if (row.size() >= 5)
        {
            // string combined = row[4];
            // for (int i = 5; i < row.size() - 1; i++)
            // {
            //     if (i == 7 || i >= 20)
            //     {
            //         continue;
            //     }
            //     combined = combined + " " + row[i];
            // }

            // Point point(pointId, combined);
            // all_points.push_back(point);
            // pointId++;
            string combined = row[0];
            for (int i =1; i < row.size() ; i++)
            {

                combined = combined + " " + row[i];
            }

            Point point(pointId, combined);
            all_points.push_back(point);
            pointId++;
            
        }
    }

    infile.close();
    cout << "\nData fetched successfully!" << endl << endl;

    if ((int)all_points.size() < K) {
        cout << "Error: Number of clusters greater than number of points." << endl;
        return 1;
    }

    int iters = 300;
    for (int i = 6; i <= K; i++)

    {
        KMeans kmeans(i, iters, output_dir,thread_num);
        kmeans.run(all_points);
    }

    auto end_time = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end_time - start_time);
    cout << "Program completed in " << duration.count() << " ms." << endl;

    return 0;
}
