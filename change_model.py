import onnx

# onnx_path = 'yolov5s_official.onnx'
onnx_path = '../yolov5-6.2/out.onnx'
onnx_model = onnx.load(onnx_path)
graph = onnx_model.graph
node = graph.node
initializer = graph.initializer


# delete Identity_2 node
# graph.node.remove(node[2])


resize_data = onnx.helper.make_tensor('onnx::Resize_418', onnx.TensorProto.FLOAT, [4], [1, 1, 2, 2])
initializer.append(resize_data)

# onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, 'out1.onnx')
