#include <iostream>
#include "MultinomialNB.h"

int main() {
    MultinomialNB classifier;

    classifier.AddTrainingSet(
        "weather", {
            {"how", "is", "the", "weather"},
            {"is", "it", "rainy", "outside"},
            {"tell", "me", "is", "it", "sunny", "outside"},
            {"what", "is", "the", "weather", "like", "for", "the", "week"},
            {"how", "cloudy", "is", "it"},
            {"should", "i", "stay", "inside", "today"},
            {"when", "will", "it", "stop", "raining"},
            {"weather", "outside", "sunny", "rainy"},
            {"can", "you", "tell", "me", "the", "weather"},
            {"raining", "weather", "hot", "cold", "temperature"},
            {"cloudy", "inside", "today", "tomorrow", "week", "day"},
        }
    );
    classifier.AddTrainingSet( 
        "grades", {
            {"what", "are", "my", "grades", "like"},
            {"how", "did", "i", "do", "on", "that", "test"},
            {"what", "is", "my", "english", "grade", "like"},
            {"how", "are", "my", "grades"},
            {"what", "did", "i", "score", "on", "that", "quiz"},
            {"did", "i", "turn", "in", "my", "homework"},
            {"grade", "grades", "score", "scores", "test", "class"},
            {"math", "english", "science", "subject", "subjects"},
        }
    );
    classifier.AddTrainingSet(
        "jokes", {
            {"can", "you", "tell", "me", "a", "joke"},
            {"give", "me", "something", "funny"},
            {"lighten", "the", "mood", "for", "me"},
            {"tell", "me", "something", "funny"},
            {"i", "want", "to", "hear", "a", "joke"},
        }
    );
    classifier.AddTrainingSet(
        "greeting", {
            {"hello", "how", "are", "you"},
            {"hello"},
            {"nice", "to", "meet", "you"},
            {"hey", "how", "is", "it", "going"},
            {"hows", "it", "going"},
            {"whats", "up"},
            {"hello", "its", "nice", "to", "meet", "you"},
            {"hey"},
            {"hello", "my", "name", "is"},
            {"hey", "im"},
            {"hello", "i", "am"},
        }
    );




    classifier.PrepareData();
    classifier.CalculateWordProbabilities();

    // std::cout << classifier.GetTrainingData()->at(0).word_count << "\n";
    // std::cout << classifier.GetTrainingData()->at(1).individual_word_counts["grades"] << "\n";
    // std::cout << classifier.GetTrainingData()->at(0).probabilities["week"] << "\n";
    // std::cout << classifier.GetTrainingData()->at(1).probabilities["grades"] << "\n";

    std::string sentence = "how are my grades looking today";
    std::string prediction = classifier.MakePrediction(sentence);
    std::cout << prediction << "\n";
    classifier.DisplayCategoryPercentages();


    // for(std::string word : classifier.GetVocabulary()) {
    //     std::cout << word << "\n";
    // }

    // std::cout << classifier.GetVocabulary().size() << "\n";
    // std::cout << classifier.GetTrainingData()->at(0).phrases[0][0] << "\n";

    return 0;
}