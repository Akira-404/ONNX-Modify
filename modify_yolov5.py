import onnx
from onnx import helper
from parse_onnx import ParseONNX

po = ParseONNX('model/yolov5s_official.onnx')
init_data = po.get_initializer()

# delete
i1, delete_node = po.find_node_with_name('Identity_2')
print(i1, delete_node)

# modify
i2, resize_node = po.find_node_with_name('Resize_143')
print(i2, resize_node)

scales = helper.make_tensor('upsample', onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
resize_node.input[2] = 'upsample'
print(resize_node)
init_data.append(scales)

# delete
po.delete(i1)
#
po.export('model/yolov5_1.onnx')
