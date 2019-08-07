#include <iostream>
#include "MultinomialNB.h"

int main() {
    MultinomialNB model;

    model.ReadInTrainingData();
    model.PrepareData();
    model.Train();

    std::string sentence = "hey hows it going, my name is drew";
    std::string prediction = model.Classify(sentence);
    std::cout << sentence << " -> ";
    std::cout << prediction << "\n\n";
    model.DisplayCategoryPercentages();

    return 0;
}