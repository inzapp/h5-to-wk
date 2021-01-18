export LD_LIBRARY_PATH="./:${LD_LIBRARY_PATH}"
./nnie_mapper_12 ./config.cfg
rm ../*.pb
rm ../*.npy
np ../*.json
#cp -avf ../model.wk ~/nfs/target_hi3519a/home/sbd.wk
#ls -lrt ~/nfs/target_hi3519a/home/
#date
