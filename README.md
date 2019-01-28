# Convert pytorch to Caffe by ONNX
This tool converts [pytorch](https://github.com/pytorch/pytorch) model to [Caffe](https://github.com/BVLC/caffe) model by [ONNX](https://github.com/onnx/onnx)  
only use for inference

### Dependencies
* python 3.5.2
* pycaffe (with python3 support)
* pytorch 1.0.0 
* onnx 1.4.1
* protobuf 3.6.1
* pyhocon 0.3.50(for config)

### How to use
To convert onnx model to caffe:
```
python convertCaffe.py --conf-path ${CONFIG}.hocon
```
### Current support operation
* Conv
* ConvTranspose
* BatchNormalization
* MaxPool
* AveragePool
* Relu
* Sigmoid
* Dropout
* Gemm (InnerProduct only)
* Add
* Mul
* Reshape
* Upsample
* Flatten

### TODO List
 - [x] Remove *Constant*, *Shape*, *Unsqueeze*, *Squeeze* ops in onnx and concatenate prototxt after removing these ops
 - [ ] *Concat* layer
     - the common usage of pytorch *view* operation, the *Concat* before the *Reshape* is redundant for caffe
 - [ ] merge batchnormization to convolution
 - [ ] merge scale to convolutionv
