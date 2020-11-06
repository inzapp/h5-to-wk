python3 h5_to_pb.py
mmconvert -sf tf -iw model.pb -df caffe -om model --inputShape=368,640,1 --inNodeName=x --dstNodeName=Identity 
rm -r __pycache__
#rm *.pb
rm *.json
rm *.npy
