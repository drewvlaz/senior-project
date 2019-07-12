// This program is a multinomial Naive Bayes classifier for 
// for text classification
// 
// The Naive Bayes classifier is based off of Bayes Theorem:
// 
//                      P(B|A)P(A)
//           P(A|B)  =  ----------
//                         P(B)
// 
// Implementation:
// 
//          P(c|X)  =  P(x1|c)P(x2|c)...P(xn|c)P(c)
// 
// Naive Bayes assumes each feature of set (X) contributes
// equally and indepently to the class (c), hence the name
// Because this is calculated for each class with a given
// feature set, the denominator remains constant and can
// therefore be ignored

#include <iostream>
#include <algorithm>    // std::find
#include <map>          // std::map
#include <vector>       // std::vector


class MultinomialNB {
private:
    struct Category {
        std::string label;
        std::vector<std::vector<std::string> > phrases;     // list of phrases containing a list of words
        std::map<std::string, double> probabilities;        // map of word to probability given the category
        std::map<std::string, int> individual_word_counts;             // map of word to number of apperances in data
    };
    std::vector<Category> m_training_data;              // contains category labels and examples
    std::vector<std::string> m_vocabulary;              // list of unique words
    int total_word_count;

public:
    MultinomialNB() {
    }

    auto GetTrainingData() { return &m_training_data; }
    auto GetVocabulary() { return m_vocabulary; }

    void AddTrainingData(Category &data) {
        m_training_data.push_back(data);
    }

    bool VocabContains(std::string word) {
        // std::find returns an iterator to element if found, 
        // else returns end of the vector in O(n) time
        return std::find(m_vocabulary.begin(), m_vocabulary.end(), word) != m_vocabulary.end();
    }

    void PrepareData() {
        for(Category &category : m_training_data) {
            for(const std::vector<std::string> &phrase : category.phrases) {
                for(const std::string &word : phrase) {
                    if(!VocabContains(word)) {
                        m_vocabulary.push_back(word);
                    }
                    ++category.individual_word_counts[word];
                    ++category.total_word_count;
                }
            }
        }
    }

    void CalculateWordProbabilities() {
        for(Category &category : m_training_data) {
            for(const std::vector<std::string> &phrase : category.phrases) {
                for(const std::string &word : phrase) {
                    category.probabilities[word] = {
                        // calculate probability of a word given a category
                        // add 1 to numerator and vocab size to denominator for smoothing
                        static_cast<double>(category.individual_word_counts[word] + 1)
                        / (category.total_word_count + m_vocabulary.size())
                    };
                }
            }
        }
    }
};

int main() {
    MultinomialNB classifier;
    // Category weather = {
    //     "weather", {
    //         {"rainny"}, 
    //         {"what", "is", "the", "weather", "like", "for", "the", "week"},
    //         {"how", "is", "the", "weather","today"}
    //     }
    // };

    // Category grades = {
    //     "grades", {
    //         {"what", "are", "my", "grades", "like"},
    //         {"how", "did", "i", "do", "on", "that", "test"},
    //         {"what", "is", "my", "english", "grade", "like"},
    //         {"how", "are", "my", "grades"},
    //         {"what", "did", "i", "score", "on", "that", "quiz"},
    //         {"did", "i", "turn", "in", "my", "homework"},
    //         {"grade", "grades", "score", "scores", "test", "class"},
    //         {"math", "english", "science", "subject", "subjects"},
    //     }
    // };
    
    // classifier.AddTrainingData(weather);
    // classifier.AddTrainingData(grades);
    // classifier.PrepareData();
    // classifier.CalculateWordProbabilities();
    // // std::cout << classifier.GetTrainingData().at(0).individual_word_counts["is"] << "\n";
    // std::cout << classifier.GetTrainingData()->at(0).total_word_count << "\n";
    // std::cout << classifier.GetTrainingData()->at(1).total_word_count << "\n";
    // std::cout << classifier.GetTrainingData()->at(1).individual_word_counts["grades"] << "\n";
    // std::cout << classifier.GetTrainingData()->at(1).probabilities["grades"] << "\n";

    // // for(std::string word : classifier.GetVocabulary()) {
    // //     std::cout << word << "\n";
    // // }

    // std::cout << classifier.GetVocabulary().size() << "\n";
    // std::cout << classifier.GetTrainingData()->at(0).phrases[0][0] << "\n";

    return 0;
}
