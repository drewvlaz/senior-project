#include <iostream>
#include <algorithm>    // std::find
#include <map>          // std::map
#include <vector>       // std::vector
#include <sstream>

std::vector<std::string> SplitString(std::string sentence) {
    std::string buffer;                 // Have a buffer string
    std::stringstream ss(sentence);       // Insert the string into a stream
    std::vector<std::string> tokens; // Create vector to hold our words

    while (ss >> buffer)
        tokens.push_back(buffer);

    return tokens;
}

std::string ToLowerCase(std::string sentence) {
    
}

int main() {

    std::vector<std::string> split = SplitString("Hello world");

    std::cout << split[0] << "\n";

    return 0;
}