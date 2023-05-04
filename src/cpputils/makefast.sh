c++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) src/cpputils.cpp src/*.hpp -o src/cpputils$(python3-config --extension-suffix)

