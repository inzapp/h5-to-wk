export LD_LIBRARY_PATH="./lib:${LD_LIBRARY_PATH}"
./nnie_mapper_12 ./config.cfg
rm ../*.pb
rm ../*.npy
rm ../*.json
cp -avf ../model.wk ~/nfs/target_hi3519a/home/tmp.wk
cp -avf ../model.wk ~/svn/ccl/model/tmp.wk
ls -lrt ~/nfs/target_hi3519a/home/
date
