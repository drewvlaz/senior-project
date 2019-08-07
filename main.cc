#include <iostream>
#include "MultinomialNB.h"

int main() {
    MultinomialNB model;

    model.ReadInTrainingData();
    model.PrepareData();
    model.Train();

    std::string sentence = "how is the weather today, i hope it is good";
    std::string prediction = model.MakePrediction(sentence);
    std::cout << sentence << " -> ";
    std::cout << prediction << "\n\n";
    model.DisplayCategoryPercentages();

    return 0;
}