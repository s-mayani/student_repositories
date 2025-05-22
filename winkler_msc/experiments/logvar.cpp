#include <iostream>
#define printvar(X) std::cout << #X << ": " << X << "\n"
int main(){
    int value = 6;
    printvar(value);
}