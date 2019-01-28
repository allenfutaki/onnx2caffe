from __future__ import print_function
from util.option import Option 
import argparse
import sys, os
sys.path.append('/home/allen/Documents/caffe/python')
import caffe
from caffe.proto import caffe_pb2
import torch
import torch.nn
import onnx
import numpy as np
caffe.set_mode_cpu()
from onnx2caffe._transformers import ConvAddFuser,ConstantsToInitializers
from onnx2caffe._graph import Graph

import onnx2caffe._operators as cvt
import onnx2caffe._weightloader as wlr
from onnx2caffe._error_utils import ErrorHandling
from collections import OrderedDict
from onnx import shape_inference
import importlib

transformers = [
    ConstantsToInitializers(),
    ConvAddFuser(),
] 

def convertToCaffe(graph, prototxt_save_path, caffe_model_save_path):

    exist_edges = []
    layers = []
    exist_nodes = []
    err = ErrorHandling()
    for i in graph.inputs:
        edge_name = i[0]
        input_layer = cvt.make_input(i)
        layers.append(input_layer)
        exist_edges.append(i[0])
        graph.channel_dims[edge_name] = graph.shape_dict[edge_name][1]


    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False

        for inp in inputs:
            if inp not in exist_edges and inp not in inputs_tensor:
                input_non_exist_flag = True
                break
        if input_non_exist_flag:
            continue

        if op_type not in cvt._ONNX_NODE_REGISTRY:
            # converter_fn = cvt._ONNX_NODE_REGISTRY['Unknown']
            print("passing unknown layer: {}".format(op_type))
            err.unsupported_op(node)
        else:
            converter_fn = cvt._ONNX_NODE_REGISTRY[op_type]
        layer = converter_fn(node,graph,err)
        if type(layer)==tuple:
            for l in layer:
                layers.append(l)
        else:
            layers.append(layer)
        outs = node.outputs
        for out in outs:
            exist_edges.append(out)

    net = caffe_pb2.NetParameter()
    for id,layer in enumerate(layers):
        layers[id] = layer._to_proto()
    net.layer.extend(layers)

    with open(prototxt_save_path, 'w') as f:
        print(net,file=f)

    caffe.set_mode_cpu()
    deploy = prototxt_save_path
    net = caffe.Net(deploy,
                    caffe.TEST)

    for id, node in enumerate(graph.nodes):
        node_name = node.name
        op_type = node.op_type
        inputs = node.inputs
        inputs_tensor = node.input_tensors
        input_non_exist_flag = False
        if op_type not in wlr._ONNX_NODE_REGISTRY:
            err.unsupported_op(node)
            continue
        converter_fn = wlr._ONNX_NODE_REGISTRY[op_type]
        converter_fn(net, node, graph, err)

    net.save(caffe_model_save_path)
    return net

def getGraph(onnx_path, pth_path):
    model = onnx.load(onnx_path)
    model = shape_inference.infer_shapes(model)
    model_graph = model.graph
    graph = Graph.from_onnx(model_graph, pth_path)
    graph = graph.transformed(transformers)
    graph.channel_dims = {}

    return graph

def cleanGraph(graph):
    clean_nodes = {}
    for id, node in enumerate(graph.nodes):
        op_type = node.op_type
        
        if op_type not in cvt._ONNX_NODE_REGISTRY:
            if len(node.parents) == 1 or len(node.children) == 1:
                clean_nodes[node.name] = node
                if len(node.input_tensors) == 0 : # no weight in this node
                    if len(node.parents) == 1:
                        if len(node.parents[0].inputs) > 0:
                            piece_head = node.parents[0] 
                        else:
                            i = 1
                            while(graph.nodes[id-i].name in clean_nodes):
                                i += 1
                            piece_head = graph.nodes[id-i]

                        if node in piece_head.children:
                            piece_head.children.remove(node)
            
                        for child in node.children:   
                            child.parents.remove(node)
                            index = child.inputs.index(node.name)
                            child.inputs.remove(node.name)
                            
                            if piece_head.name in child.inputs:
                                continue
                            piece_head.children.append(child)
                            child.parents.append(piece_head)
                            child.inputs.insert(index, piece_head.name)
                    else:
                        if len(node.inputs) == 0:
                            i = 1
                            while(graph.nodes[id-i].name in clean_nodes):
                                i += 1
                            node.parents.append(graph.nodes[id-i])
                            node.inputs.append(graph.nodes[id-i])

                        piece_tail = node.children[0]
                        piece_tail.parents.remove(node)
                        piece_tail.inputs.remove(node.name)
                        for parent in node.parents:
                            if len(parent.inputs) == 0:
                                continue
                            parent.children.remove(node)

                            piece_tail.parents.append(parent)
                            parent.children.append(piece_tail)
                            piece_tail.inputs.insert(0,parent.name)
            else:
                print("multiple inputs and multiple outputs are confused ...")
                sys.exit(1)

    for key in clean_nodes.keys():
        graph.nodes.remove(clean_nodes[key])
    return graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pytorch to Caffe")
    parser.add_argument('--conf-path', type=str, metavar='conf_path',
                        help='configuration path')
    args = parser.parse_args()

    config = Option(args.conf_path)

    graph = getGraph(config.onnxmodel)
    cleanGraph(graph)
    convertToCaffe(graph, config.prototxt, config.caffemodel)



















