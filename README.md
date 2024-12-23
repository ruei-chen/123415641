# Parallelizing K-Means Clustering for Scalable Customer Segmentation

## Evaluation platform 

Ubuntu 22.04 with
Intel(R) Core(TM) i5-13500 CPU
@ 2.50GHz processors and
32GB RAM and
GeForce RTX 4060 8GB.

Enter the specified folder
```
$ cd parallel_kmeans
```

## Dataset : marketing_campaign.csv

Content
AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise<br>
AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise<br>
AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise<br>
AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise<br>
AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise<br>
Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise<br>
Complain - 1 if customer complained in the last 2 years<br>
DtCustomer - date of customer’s enrolment with the company<br>
Education - customer’s level of education<br>
Marital - customer’s marital status<br>
Kidhome - number of small children in customer’s household<br>
Teenhome - number of teenagers in customer’s household<br>
Income - customer’s yearly household income<br>
MntFishProducts - amount spent on fish products in the last 2 years<br>
MntMeatProducts - amount spent on meat products in the last 2 years<br>
MntFruits - amount spent on fruits products in the last 2 years<br>
MntSweetProducts - amount spent on sweet products in the last 2 years<br>
MntWines - amount spent on wine products in the last 2 years<br>
MntGoldProds - amount spent on gold products in the last 2 years<br>
NumDealsPurchases - number of purchases made with discount<br>
NumCatalogPurchases - number of purchases made using catalogue<br>
NumStorePurchases - number of purchases made directly in stores<br>
NumWebPurchases - number of purchases made through company’s web site<br>
NumWebVisitsMonth - number of visits to company’s web site in the last month<br>
Recency - number of days since the last purchase<br>

## Execution process

Serial : make & implement
```
$ g++ final_serial.cc -o kmeans
$ ./kmeans marketing_campaign.csv 8 output
```
Openmp : make & implement
```
$ g++ -g final_openmp1.cpp -o kmeans -fopenmp
$ ./kmeans marketing_campaign.csv 8 output
$ g++ -g final_openmp2.cpp -o kmeans -fopenmp
$ ./kmeans marketing_campaign.csv 8 output
$ g++ -g final_openmpsimd.cpp -o kmeans -fopenmp
$ ./kmeans marketing_campaign.csv 8 output
```
Pthreads : make & implement
```
$ g++ -g pthread_test.cc -o kmeans -pthread
$ ./kmeans marketing_campaign.csv 8 output N (where N is the number of thread)
```
CUDA : make & implement
```
$ nvcc -g final.cu -o kmeans
$ ./kmeans marketing_campaign.csv 8 output
```

---------------------------------------------------------------------------------------------------------------------

Due to the original dataset containing too few entries, the time saved by parallelizing specific parts appeared relatively insignificant in the overall execution. Therefore, we generated a dataset with 100,000 entries to validate the effectiveness of our parallelization efforts.

## Dataset : 1.csv
A total of 100,000 pieces of data with 10 fields

## Execution process

Serial : make & implement
```
$ g++ -g final_serial_1.cc -o kmeans
$ ./kmeans 1.csv 6 output
```
Openmp : make & implement
```
$ g++ -g final_openmp1_1.cpp -o kmeans -fopenmp
$ ./kmeans 1.csv 6 output
$ g++ -g final_openmp2_1.cpp -o kmeans -fopenmp
$ ./kmeans 1.csv 6 output
$ g++ -g final_openmpsimd_1.cpp -o kmeans -fopenmp
$ ./kmeans 1.csv 6 output
```
Pthreads : make & implement
```
$ g++ -g pthread_test_1.cc -o kmeans -pthread
$ ./kmeans 1.csv 6 output N (where N is the number of thread)
```
CUDA : make & implement
```
$ nvcc -g final_1.cu -o kmeans
$ ./kmeans 1.csv 6 output
```
