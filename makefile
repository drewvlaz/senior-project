CXXFLAGS = -ljsoncpp

main: main.o MultinomialNB.o
	g++ $(CXXFLAGS) main.o MultinomialNB.o -o main

main.o: main.cc
	g++ $(CXXFLAGS) -c main.cc

MultinomialNB.o: MultinomialNB.cc MultinomialNB.h
	g++ $(CXXFLAGS) -c MultinomialNB.cc

clean:
	rm *.o main