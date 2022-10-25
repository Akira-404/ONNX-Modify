import onnx
import os


class ParseONNX:
    def __init__(self, onnx_path: str):
        assert os.path.exists(onnx_path), f'{onnx_path} is not found.'

        self._onnx_path = onnx_path
        self._model = None
        self._graph = None
        self._nodes = None
        self._input = None
        self._output = None
        self._initializer = None
        self._load()

    def _load(self):
        self._model = onnx.load(self._onnx_path)
        self._graph = self._model.graph
        self._input = self._graph.input
        self._output = self._graph.output
        self._nodes = self._graph.node
        self._initializer = self._graph.initializer

    def get_input(self) -> onnx.ValueInfoProto:
        return self._input

    def get_output(self) -> onnx.ValueInfoProto:
        return self._output

    def get_model(self) -> onnx.ModelProto:
        return self._model

    def get_graph(self) -> onnx.GraphProto:
        return self._graph

    def get_initializer(self) -> onnx.TensorProto:
        return self._initializer

    def get_nodes(self) -> onnx.NodeProto:
        return self._nodes

    def find_node_with_type(self, op_type: str = None) -> list:
        indexes = []
        for i, node in enumerate(self._nodes):
            if node.op_type == op_type:
                indexes.append(i)
        return indexes

    def find_node_with_name(self, name: str = None):
        for i, node in enumerate(self._nodes):
            if node.name == name:
                return i, node
        return -1, None

    def find_initializer_with_name(self, name: str = None) -> onnx.TensorProto:
        for i, data in enumerate(self._initializer):
            if data.name == name:
                return data
        return None

    def export(self, path: str, check: bool = True):
        if check:
            onnx.checker.check_model(self._model)
        onnx.save(self._model, path)
        print(f'onnx model save to {path}.')

    def delete(self, i: int):
        self._nodes.remove(self._nodes[i])


if __name__ == '__main__':
    onnx_path = 'model/resize_linear.onnx'
    parseonnx = ParseONNX(onnx_path)
    init_data = parseonnx.get_initializer()
    print(init_data)
