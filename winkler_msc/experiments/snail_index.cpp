#include "libmorton/include/libmorton/morton.h"
#include <iostream>
#include <cassert>
#include <algorithm>
#include <format>
using std::min;
using std::max;
using u32 = unsigned int;
constexpr u32 extenz = 12;
void call(u32 i, u32 j, u32 k){
    assert(
        i == 0 ||
        j == 0 ||
        k == 0 ||
        i == extenz - 1 ||
        j == extenz - 1 ||
        k == extenz - 1
    );
    printf("%u,%u,%u\n", i, j, k);
}
int main(){
    u32 count = extenz * extenz * extenz - (extenz - 2) * (extenz - 2) * (extenz - 2);
    u32 sq = extenz * extenz;
    u32 sqh = extenz * extenz - (extenz - 2) * (extenz - 2);
    for(u32 i = 0;i < count;i++){
        if(i < sq){
            call(i % extenz, i / extenz, 0);
        }
        else if(i >= count - sq){
            u32 i2 = i + sq - count;
            call(i2 % extenz, i2 / extenz, extenz - 1);
        }
        else{
            const int val = (i - sq) % sqh;
            const int side = extenz - 1;

            int i1 = min(side, val);
            int i2 = min(side, max(0, val - side));
            int i3 = min(side, max(0, val - side * 2));
            int i4 = min(side, max(0, val - side * 3));
            

            call(i1 - i3, i2 - i4, (i - sq) / sqh + 1);
        }
    }
}
