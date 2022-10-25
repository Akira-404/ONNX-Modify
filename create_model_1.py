import onnx

from onnx import helper
from onnx import TensorProto

# 构造用于描述张量的ValueInfo对象
a = helper.make_tensor_value_info(name='a', elem_type=TensorProto.FLOAT, shape=[10, 10])
x = helper.make_tensor_value_info(name='x', elem_type=TensorProto.FLOAT, shape=[10, 10])
b = helper.make_tensor_value_info(name='b', elem_type=TensorProto.FLOAT, shape=[10, 10])

output = helper.make_tensor_value_info(name='output', elem_type=TensorProto.FLOAT, shape=[10, 10])

# 构造节点对象
mul = helper.make_node(name='Mul', op_type='Mul', inputs=['a', 'x'], outputs=['c'])
add = helper.make_node(name='Add', op_type='Add', inputs=['c', 'b'], outputs=['output'])

# 构造图对象
graph = helper.make_graph(nodes=[mul, add], name='linear', inputs=[a, x, b], outputs=[output])

# 构造模型对象
model = helper.make_model(graph=graph)

onnx.checker.check_model(model)
print(model)
onnx.save(model, 'model/linear.onnx')
node = model.graph.node
print(node)
