# BalancingScale
Ultra Low Latency production standard code for High Frequency Trading.
This code is to balance unbalanced scale as mentioned in the attahed problem document Balancing_Scale_Problem.pdf.

Important Files:

A) Problem document: Balancing_Scale_Problem.pdf
B) Solution in Ultra Low Latency C++: BalancingScale.cpp
C) Test Data Input/Output generator: BalancingScale_Test_Data_Generator.py
---------------------------------------------------------------------------------

B) BalancingScale.cpp :
----------------------
1. Code is written to achieve best possible Ultra Low Latency taking care all possible considerations like - Memory pool, pipelining, cache line, prefetch etc. 
2. Code is written in production standard.
3. Older compiler may not support certain modern C++ library. So, pls install the library if compilation fails. For example: I had to install llvm library to support to_char/from_char()
4. Use correct command/flags/path to compile as per your machine. For example, I  compile like this on my MacBook :

> clang++ -std=c++20 -stdlib=libc++ -I/opt/homebrew/opt/llvm/include -L/opt/homebrew/opt/llvm/lib -Wall -O3 ./BalancingScale.cpp -o ./BalancingScale/BalancingScale 


C) BalancingScale_Test_Data_Generator.py :
----------------------------------------
1. BalancingScale_Test_Data_Generator.py is simple python tool to generate random test data. It will generate input and expected output test data which you can use to test the C++ code.
2. Command to run:
   > python3 BalancingScale_Data_Generator.py

===========================================================================================================================================
Contact: If any issue or suggestion, kindly contact at sheikhazad2@yahoo.com



