import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

# input1*input2->resize=input12
# input12+input3=output

# inputs
input1 = helper.make_tensor_value_info(name='input1', elem_type=TensorProto.FLOAT, shape=[6, 6])
input2 = helper.make_tensor_value_info(name='input2', elem_type=TensorProto.FLOAT, shape=[6, 6])
input3 = helper.make_tensor_value_info(name='input3', elem_type=TensorProto.FLOAT, shape=[3, 3])

# outputs
output = helper.make_tensor_value_info(name='output', elem_type=TensorProto.FLOAT, shape=[3, 3])

# nodes
mul = helper.make_node(name='mul_0', op_type='Mul', inputs=['input1', 'input2'], outputs=['input_12'])

scales = helper.make_tensor('downsample_scales', onnx.TensorProto.FLOAT, [4], [1, 1, 0.5, 0.5])
downsample = helper.make_node(name='downsample', op_type='Resize',
                              inputs=['input_12', '', 'downsample_scales'],
                              outputs=['resize_input_12'],
                              cubic_coeff_a=-0.75,
                              mode='nearest',
                              nearest_mode='floor')

add = helper.make_node(name='add_0', op_type='Add', inputs=['resize_input_12', 'input3'], outputs=['output'])

initializer = [scales]

graph = helper.make_graph(nodes=[mul, downsample, add],
                          name='resize_linear',
                          inputs=[input1, input2, input3],
                          outputs=[output],
                          initializer=initializer)

model = helper.make_model(graph)
onnx.checker.check_model(model)
onnx.save(model, 'model/resize_linear.onnx')
