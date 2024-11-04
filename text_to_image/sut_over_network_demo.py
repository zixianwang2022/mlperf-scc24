# Copyright 2019 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================


"""
Python demo showing how to use the MLPerf Inference load generator bindings over the network.
This part of the demo runs the "demo SUT" which is connected over the network to the LON node.
A corresponding "demo LON node" with the demo test is implemented in py_demo_server_lon.py.

The SUT is implemented using a Flask server, with dummy implementation of the inference processing.
Two endpoints are exposed:
- /predict/ : Receives a query (e.g., a text) runs inference, and returns a prediction.
- /getname/ : Get the name of the SUT.

The current implementation is a dummy implementation, which does not use 
a real DNN model, batching, or pre/postprocessing code,
but rather just returns subset of the input query as a response,
Yet, it illustrates the basic structure of a SUT server.




Run on each inference node:
python sut_over_network_demo.py --port 8888 --node <N1 or N2>

"""
from main import Item
from main import QueueRunner
from main import get_args
from main import get_backend


import argparse
from flask import Flask, request, jsonify

import subprocess


app = Flask(__name__)



node = ""
def preprocess(query):
    """[SUT Node] A dummy preprocess."""
    # Here may come for example batching, tokenization, resizing, normalization, etc.
    response = query
    return response


def dnn_model(query):
    """[SUT Node] A dummy DNN model."""
    # Here may come for example a call to a dnn model such as resnet, bert, etc.
    response = query
    return response


def postprocess(query):
    """[SUT Node] A dummy postprocess."""
    # Here may come for example a postprocessing call, e.g., NMS, detokenization, etc.
    response = query
    return response


# needs to receive list of prompts [prompt1, prompt2,]
@app.route('/predict/', methods=['POST'])
def predict():
    """Receives a query (e.g., a text) runs inference, and returns a prediction."""
    query = request.get_json(force=True)['query']
    result = postprocess(dnn_model(preprocess(query)))
    return jsonify(result=result)
    
    


@app.route('/getname/', methods=['POST', 'GET'])
def getname():
    """Returns the name of the SUT."""
    return jsonify(name=f'Demo SUT (Network SUT) node' + (' ' + node) if node else '')


if __name__ == '__main__':
    # Get the IP address of current node
    command = "ip addr show eth0 | grep 'inet ' | awk '{print $2}' | cut -d'/' -f1"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    ipaddr = result.stdout.strip()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--node', type=str, default="")
    parser.add_argument('--ipaddr', type=str, default=ipaddr)
    args = parser.parse_args()
    node = args.node
    app.run(debug=False, port=args.port, host=args.ipaddr)
