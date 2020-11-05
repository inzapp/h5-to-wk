python3 h5_to_pb.py
cp -afv log/model.pb model.pb
mmconvert -sf tf -iw model.pb -df caffe -om model --inputShape=368,640,1 --inNodeName=x --dstNodeName=Identity --dump_tag=SERVING
rm -r __pycache__
