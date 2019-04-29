RNN: RNN.cpp activation.o
	g++ -o RNN RNN.cpp -L/usr/local/lib -lcnpy -lz -lsleef --std=c++11 -mavx
