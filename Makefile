all:
	swig3.0 -c++ -python ploterm.i
	g++ -fPIC -shared -std=c++11 ploterm.cpp ploterm_wrap.cxx -I/usr/include/python2.7 -I/usr/include/x86_64-linux-gnu/python2.7 -lpthread -ldl -lutil -lm -lpython2.7 -o _ploterm.so
