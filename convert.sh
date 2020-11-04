python3 h5_to_pb.py
mmconvert -sf tf -iw model.pb -df caffe -om model --inputShape=386,640,1 --inNodeName=x --dstNodeName=Identity
rm -r __pycache__
