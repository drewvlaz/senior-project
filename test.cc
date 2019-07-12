#include <iostream>
#include <map>          // For std::map
#include <vector>       // For std::vector

struct Category {
    std::string title;
    std::vector<std::vector<std::string>> phrases;
};

int main() {

    std::vector<Category> m_training_data;
    m_training_data.resize(10);

    m_training_data[0].title = "greetings";
    m_training_data[0].phrases = {{ "hello" }, { "how are you" }};

    m_training_data[1] = {"weather", {{ "rainny" }, {"what", "is", "the", "weather", "like", "for", "the", "week"}}};

    for(std::string word : m_training_data[1].phrases[1]) {
        std::cout << word << "\n";
    }

    std::cout << m_training_data[1].phrases[1].size() << "\n\n";

    return 0;
}