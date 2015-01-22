#include <iostream>

using namespace std;

int main(void)
{
    float f;
    while(cin >> f)
        cout.write((char *)&f,sizeof(float));
    return 0;
}
