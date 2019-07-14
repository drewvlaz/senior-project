main: main.o MultinomialNB.o
	g++ main.o MultinomialNB.o -o main

main.o: main.cc
	g++ -c main.cc

MultinomialNB.o: MultinomialNB.cc MultinomialNB.h
	g++ -c MultinomialNB.cc

clean:
	rm *.o main