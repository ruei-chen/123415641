# Parallelizing K-Means Clustering for Scalable Customer Segmentation

實驗環境 

Ubuntu 22.04 with
Intel(R) Core(TM) i5-13500 CPU
@ 2.50GHz processors and
32GB RAM and
GeForce RTX 4060 8GB.

```
$ cd parallel_kmeans
```
Serial
```
$ g++ final_serial.cc -o kmeans
$ ./kmeans marketing_campaign.csv 8 output
```
Openmp
```
$ g++ final_openmp1.cpp -o kmeans -fopenmp
$ ./kmeans marketing_campaign.csv 8 output
$ g++ final_openmp2.cpp -o kmeans -fopenmp
$ ./kmeans marketing_campaign.csv 8 output
$ g++ final_openmpsimd.cpp -o kmeans -fopenmp
$ ./kmeans marketing_campaign.csv 8 output
```
Pthreads
```

```
CUDA
```
$ nvcc final.cu -o kmeans
$ ./kmeans marketing_campaign.csv 8 output
```
