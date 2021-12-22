# -mavx is to support avx instructions
g++ -Wall optimization_challenge.cpp -o optimization_challenge -O3 -lstdc++ -lm -flto -Wall -march=native -mavx -std=c++17
./optimization_challenge
