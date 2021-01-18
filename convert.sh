export PATH=/home/kali/.local/bin:$PATH
mmconvert -sf tf -iw model.h5 -df caffe -om model --inNodeName=x --dstNodeName=Identity
