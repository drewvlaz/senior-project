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
#include <sstream>      // std::stringstream

class MultinomialNB {
private:
    struct Category {
        std::string label;                                  // identifies category
        std::vector<std::vector<std::string> > phrases;     // vector of phrases containing a list of words
        std::map<std::string, double> probabilities;        // map of word to probability given the category
        std::map<std::string, int> bag_of_words;            // map of word to number of apperances in data
        int total_word_count;                               // total word count from all phrases
    };
    std::vector<Category> m_training_data;                  // contains category labels and examples for training
    std::vector<std::string> m_vocabulary;                  // vector of unique words across all categories
    std::vector<double> m_category_probabilities;           // probability input is of each category

public:
    MultinomialNB() {}

    auto GetTrainingData() { return &m_training_data; }
    auto GetVocabulary() { return m_vocabulary; }
    auto GetCategoryProbabilities() { return m_category_probabilities; }

    void AddTrainingSet(std::string label, std::vector<std::vector<std::string> > phrases) {
        m_training_data.push_back({label, phrases});
    }

    void PrepareData() {
        for(Category &category : m_training_data) {
            for(const std::vector<std::string> &phrase : category.phrases) {
                for(const std::string &word : phrase) {
                    if(!VocabContains(word)) {
                        m_vocabulary.push_back(word);
                    }
                    ++category.bag_of_words[word];
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
                        // add 1 to numerator and add vocab size to denominator for laplace smoothing
                        static_cast<double>(category.bag_of_words[word] + 1)
                        / (category.total_word_count + m_vocabulary.size())
                    };
                }
            }
        }
    }

    std::string MakePrediction(std::string sentence) {
        std::vector<std::string> split_string {Split(sentence)};
        m_category_probabilities.resize(m_training_data.size());

        for(int i=0; i<m_training_data.size(); ++i) {
            // vectors initialize to 0, since multiplying, set it to 1
            m_category_probabilities.at(i) = 1;
            for(const std::string &word: split_string) {
                // check to see if training data of a category contains the target word
                if(m_training_data.at(i).bag_of_words[word]) {
                    // P(c|X)  *=  P(x1|c)P(x2|c)...P(xn|c)
                    // multiply probability of the target word given the category
                    m_category_probabilities.at(i) *= m_training_data.at(i).probabilities[word];
                }
                else {
                    m_category_probabilities.at(i) *= static_cast<double>(1) / m_vocabulary.size();
                }
            }
            // P(c|X) *= P(c)
            // ignoring denominator because it will be the same for all categories
            m_category_probabilities[i] *= m_training_data.at(i).phrases.size();
        }
        return m_training_data.at(Max(m_category_probabilities)).label;
    }
    
    void DisplayCategoryPercentages() {
        double sum {0};
        for(double probability : m_category_probabilities) {
            sum += probability;
        }

        for(int i=0; i<m_category_probabilities.size(); ++i) {
            double percentage = m_category_probabilities.at(i) / sum * 100;
            std::cout << m_training_data.at(i).label << " " << percentage << "\n";
        }
    }

    bool VocabContains(std::string word) {
        // std::find returns an iterator to element if found, 
        // else returns end of the vector in O(n) time
        return std::find(
            m_vocabulary.begin(),
            m_vocabulary.end(),
            word
        ) != m_vocabulary.end();
    }

    std::vector<std::string> Split(std::string sentence) {
        std::string buffer;                     // buffer string
        std::stringstream stream {sentence};    // insert string into a stream
        std::vector<std::string> tokens;        // vector to hold our words
        while (stream >> buffer){
            tokens.push_back(buffer);
        }
        return tokens;
    }

    int Max(std::vector<double> values) {
        double max {values.at(0)};
        double num;
        int index;
        for(int i=0; i<values.size(); ++i) {
            num = values.at(i);
            if(num > max) {
                max = num;
                index = i;
            }
        }
        return index;
    }
};

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

    // std::cout << classifier.GetTrainingData()->at(0).total_word_count << "\n";
    // std::cout << classifier.GetTrainingData()->at(1).individual_word_counts["grades"] << "\n";
    // std::cout << classifier.GetTrainingData()->at(0).probabilities["week"] << "\n";
    // std::cout << classifier.GetTrainingData()->at(1).probabilities["grades"] << "\n";

    std::string sentence = "hello my name is drew";
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