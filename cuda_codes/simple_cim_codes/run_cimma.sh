
cd ../../ && source setup_environment release && make && cd -
make && make ptx
./cimma 128 128 128 > output.log
