export PATH=/home/kali/.local/bin:$PATH
mmconvert -sf tf -iw model.h5 -df caffe -om model --inputShape=368,640,1 --inNodeName=x --dstNodeName=Identity 
#mmconvert -sf tf -iw model.h5 -df caffe -om model --inNodeName=x --dstNodeName=Identity
