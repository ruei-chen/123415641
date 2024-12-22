kmeans: pthread_test.cc
	g++ -g pthread_test.cc -o kmeans -pthread 

clean:
	rm kmeans


# kmeans: final_serial.cc
# 	g++ -g final_serial.cc -o kmeans -pthread 

# clean:
# 	rm kmeans
