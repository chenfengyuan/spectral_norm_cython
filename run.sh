#!/bin/bash

cd go;
go build -o spectralnorm.run;
cd ..;
python setup.py build_ext --inplace
time python spectral_norm.py 5500
time ./go/spectralnorm.run
