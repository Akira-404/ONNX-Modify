
ModelProto	:定义了整个网络的模型结构  
GraphProto	:定义了模型的计算逻辑，包含了构成图的节点，这些节点组成了一个有向图结构  
NodeProto	:定义了每个OP的具体操作  
ValueInfoProto	:序列化的张量，用来保存weight和bias  
TensorProto	:定义了输入输出形状信息和张量的维度信息  
AttributeProto	:定义了OP中的具体参数，比如Conv中的stride和kernel_size等  
