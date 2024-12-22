# Parallelizing K-Means Clustering for Scalable Customer Segmentation

Evaluation platform 

Ubuntu 22.04 with
Intel(R) Core(TM) i5-13500 CPU
@ 2.50GHz processors and
32GB RAM and
GeForce RTX 4060 8GB.

Dataset : marketing_campaign.csv

Content
AcceptedCmp1 - 1 if customer accepted the offer in the 1st campaign, 0 otherwise
AcceptedCmp2 - 1 if customer accepted the offer in the 2nd campaign, 0 otherwise
AcceptedCmp3 - 1 if customer accepted the offer in the 3rd campaign, 0 otherwise
AcceptedCmp4 - 1 if customer accepted the offer in the 4th campaign, 0 otherwise
AcceptedCmp5 - 1 if customer accepted the offer in the 5th campaign, 0 otherwise
Response (target) - 1 if customer accepted the offer in the last campaign, 0 otherwise
Complain - 1 if customer complained in the last 2 years
DtCustomer - date of customer’s enrolment with the company
Education - customer’s level of education
Marital - customer’s marital status
Kidhome - number of small children in customer’s household
 Teenhome - number of teenagers in customer’s household
 Income - customer’s yearly household income
MntFishProducts - amount spent on fish products in the last 2 years
MntMeatProducts - amount spent on meat products in the last 2 years
MntFruits - amount spent on fruits products in the last 2 years
MntSweetProducts - amount spent on sweet products in the last 2 years
MntWines - amount spent on wine products in the last 2 years
MntGoldProds - amount spent on gold products in the last 2 years
NumDealsPurchases - number of purchases made with discount
NumCatalogPurchases - number of purchases made using catalogue
NumStorePurchases - number of purchases made directly in stores
NumWebPurchases - number of purchases made through company’s web site
NumWebVisitsMonth - number of visits to company’s web site in the last month
Recency - number of days since the last purchase

進入指定folder
```
$ cd parallel_kmeans
```
Serial : make & implement
```
$ g++ final_serial.cc -o kmeans
$ ./kmeans marketing_campaign.csv 8 output
```
Openmp : make & implement
```
$ g++ final_openmp1.cpp -o kmeans -fopenmp
$ ./kmeans marketing_campaign.csv 8 output
$ g++ final_openmp2.cpp -o kmeans -fopenmp
$ ./kmeans marketing_campaign.csv 8 output
$ g++ final_openmpsimd.cpp -o kmeans -fopenmp
$ ./kmeans marketing_campaign.csv 8 output
```
Pthreads : make & implement
```

```
CUDA : make & implement
```
$ nvcc final.cu -o kmeans
$ ./kmeans marketing_campaign.csv 8 output
```
