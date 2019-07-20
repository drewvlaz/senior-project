#include <iostream>
#include "MultinomialNB.h"

int main() {
    MultinomialNB model;

    model.AddTrainingData(
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
    model.AddTrainingData( 
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
    model.AddTrainingData(
        "jokes", {
            {"can", "you", "tell", "me", "a", "joke"},
            {"give", "me", "something", "funny"},
            {"lighten", "the", "mood", "for", "me"},
            {"tell", "me", "something", "funny"},
            {"i", "want", "to", "hear", "a", "joke"},
        }
    );
    model.AddTrainingData(
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
    model.AddTrainingData(
        "search", {
            "can you look up",
            "google this please",
            "what is the definition of",
            "look up this for me",
            "hey google what is",
            "i want to search the internet",
            "look up online",
            "google online definition internet information",
        }
    );

    model.PrepareData();
    model.Train();

    std::string sentence = "how is the weather today";
    std::string prediction = model.MakePrediction(sentence);
    std::cout << sentence << " -> ";
    std::cout << prediction << "\n\n";
    model.DisplayCategoryPercentages();

    return 0;
}