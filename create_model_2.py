import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto

if __name__ == '__main__':
    X = helper.make_tensor_value_info(name='X', elem_type=TensorProto.FLOAT, shape=[3, 2])
    pads = helper.make_tensor_value_info(name='pads', elem_type=TensorProto.FLOAT, shape=[1, 4])

    value = helper.make_tensor_value_info(name='value', elem_type=AttributeProto.FLOAT, shape=[1])

    Y = helper.make_tensor_value_info(name='Y', elem_type=TensorProto.FLOAT, shape=[3, 4])

    node_def = helper.make_node(
        op_type='Pad',  # node name
        inputs=['X', 'pads', 'value'],  # inputs
        outputs=['Y'],  # outputs
        mode='constant'  # attributes
    )

    graph_def = helper.make_graph(
        nodes=[node_def],
        name='test-model',
        inputs=[X, pads, value],
        outputs=[Y]
    )

    model_def = helper.make_model(graph=graph_def, producer_name='onnx-example')

    print(f'the model is :{model_def}')
    onnx.checker.check_model(model_def)
    print('the model is checked!')
    onnx.save(model_def, 'test_mode.onnx')
