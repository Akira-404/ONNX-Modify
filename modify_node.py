import onnx

from parse_onnx import ParseONNX

onnx_model_path = 'model/resize_linear.onnx'
po = ParseONNX(onnx_model_path)

nodes = po.get_nodes()
init_data = po.get_initializer()

# downsample->upsample
i, node = po.find_node_with_name('downsample')
print(node)
node.name = 'upsample'

data = po.find_initializer_with_name('downsample_scales')
print(data)
print(data.float_data)
data.float_data[0] = 1
data.float_data[1] = 1
data.float_data[2] = 2
data.float_data[3] = 2
print(data.float_data)
po.export('model/upsample.onnx')
