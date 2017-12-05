import math, json, csv;
import time, itertools, copy;
import random;
from parse import parse;

cdef class NeuralNetwork:
	cdef object layers;
	cdef dict settings, functionParams;
	cdef float learnparam;
	
	def __init__(self, dict networkRepre):
		"""networkRepre: dictionary representing the neural network topology
		
		Example of a network representation can be acquired with function neuralNet.baseNetwork()"""

		if "settings" not in networkRepre:
			raise Exception("Network representation has to contain network settings")
		if "layers" not in networkRepre:
			raise Exception("Network representation has to contain network layers")
		if len(networkRepre["layers"]) < 2:
			raise Exception("Network representation has to contain at least 2 layers!");
		
		#save network settings and set function to default if no other is specified
		self.settings = networkRepre["settings"];
		self.learnparam = self.settings["learnparam"];
		if "function" not in self.settings:
			self.settings["function"] = {"name":"sigmoid", "alpha":1.0};
		
		#set activational function and it's derivation
		self.functionParams = copy.deepcopy(self.settings["function"]);
		if self.functionParams["name"] == "sigmoid":
			self.functionParams["activate"] = self.sigmoid;
			self.functionParams["derivate"] = self.dsigmoid;
		elif self.functionParams["name"] == "tanh":
			self.functionParams["activate"] = math.tanh;
			self.functionParams["derivate"] = self.dtanh;
		else:
			self.functionParams["activate"] = self.sigmoid;
			self.functionParams["derivate"] = self.dsigmoid;
		
		#create layers
		self.layers = [Layer(self, layer) for layer in networkRepre["layers"]];
	
	def getRepre(self):
		"""Returns the representation of a neural network as a dictionary"""
		
		return {"settings":self.settings, "layers":[layer.getRepre() for layer in self.layers]};
	
	def getCompactRepre(self):
		"""Returns the compact representation of a neural network as a dictionary
		
		Default values are omitted from the representation"""
		
		return {"settings":self.settings, "layers":[layer.getCompactRepre() for layer in self.layers]};

	def getLayers(self):
		"""Returns a list containing instances of Layer objects in the network
		
		List represents layers in the network"""
		
		return self.layers;
	
	def getLearnParam(self):
		"""returns a value of learning parameter"""
		
		return self.learnparam;
	
	def sigmoid(self, float x):
		"""x: value
		
		returns the value of a sigmoid function for x"""
		
		cdef float alpha = self.functionParams["alpha"] if "alpha" in self.functionParams else 1.0;
		return 1/(1+math.exp(-x*alpha));
	
	def dsigmoid(self, float x):
		"""x: value
		
		returns the value of a derivated sigmoid function for x"""
		
		cdef float alpha = self.functionParams["alpha"] if "alpha" in self.functionParams else 1.0;
		cdef float m = 1+math.exp(-x*alpha);
		return (alpha/m)-(alpha/(m**2));
	
	def dtanh(self, float x):
		"""x: value
		
		returns the value of a derivated tanh function for x"""
		
		return 1-(math.tanh(x)**2);
	
	def activate(self, float x):
		"""x: value
		
		returns the value of a activational function for x"""
		
		return self.functionParams["activate"](x);
	
	def derivate(self, float x):
		"""x: value
		
		returns the value of a derivated activational function for x"""
		
		return self.functionParams["derivate"](x);
	
	def out(self, float x):
		"""x: value
		
		returns the value of an output function."""
		
		return x;

	def eval(self, list inputs):
		"""inputs: list representing inputs to the network
		
		returns the output of a neural network for specific input"""
		
		cdef int i;
		cdef object neuron;
		
		#set inputs to variables on the input layer
		i = 0;
		for neuron in self.layers[0].getNeurons():
			if type(neuron) == Variable:
				if i >= len(inputs):
					raise Exception("too few inputs");
				
				neuron.setValue(inputs[i]);
				i += 1;
		
		#outputs on input layer
		cdef object outputs = [[neuron.getValue() for neuron in self.layers[0].getNeurons()]];
		cdef int size = len(self.layers);
		
		#propagate inputs forward
		for i in range(1, size):
			outputs.append(self.layers[i].eval(outputs));

		return outputs[-1];

	def calcErrors(self, list result, list expected):
		"""result: list representing the actual output of the network
		expected: list representing expected output of the network
		
		calculates and propagates the error based on the expected and actual output of the network"""
		
		if len(result) != len(expected):
			raise Exception("length of results is not the same as the expected results");

		cdef object layer;
		cdef object neuron;
		
		#reset errors on neurons
		for layer in self.layers:
			for neuron in layer.getNeurons():
				if type(neuron) == Neuron:
					neuron.setError(0.0);
		
		cdef int i;
		cdef int size = len(self.layers[-1].getNeurons())
		
		#set errors on output layer
		for i in range(0, size):
			self.layers[-1].getNeurons()[i].setError((expected[i]-result[i])*self.derivate(self.layers[-1].getNeurons()[i].getInput()));

		#propagate errors backwards
		size = len(self.layers);
		for i in range(1, size):
			self.layers[-i].propagateError();

		#modify weights
		for i in range(1, size):
			self.layers[i].modifyWeights();

	def train(self, list trainingSet):
		"""trainingSet: list of rows representing training set, every row contains 2 lists: inputs and outputs of the neural network
		
		calculates the errors and adjusts weights based on the training set
		Example: NeuralNetwork.train([[[1, 1], [1]], [[1, 0], [1]], [[0, 1], [1]], [[0, 0], [0]]])"""
		
		cdef object row;
		cdef object result;
		
		#calculate errors & modify weights for every row in the training set
		for row in trainingSet:
			result = self.eval(row[0]);
			self.calcErrors(result, row[1]);
		
cdef class Layer:
	cdef object network;
	cdef object neurons;
	
	def __init__(self, object net, list layerRepre):
		"""net: instance of a NeuralNetwork object
		
		layerRepre: list containing dictionaries, where every dictionary is a neuron with specific attributes
		creates a Layer object in the network with representation specified by layerRepre"""
		
		#init & create list of neurons in the layer
		self.network = net;
		self.neurons = [Neuron(self, repre) if "synapses" in repre and len(repre["synapses"]) > 0 else Variable(repre) if "input" not in repre or repre["input"] == None else Constant(repre) for repre in layerRepre];
	
	def getRepre(self):
		"""returns a list containing representations of neurons in the layer"""
		
		return [neuron.getRepre() for neuron in self.neurons];
	
	def getCompactRepre(self):
		"""returns a list containing compact representations of neurons in the layer
		
		default values are omitted"""
		
		return [neuron.getCompactRepre() for neuron in self.neurons];

	def getNetwork(self):
		"""returns the instance of the network the layer is in"""
		
		return self.network;

	def getNeurons(self):
		"""returns a list of Neuron objects in the layer"""
		
		return self.neurons;

	def eval(self, list outputs):
		"""outputs: 2d list of outputs in the network
		output[L][N]
	
		calculates the outputs of neurons in the layer based on the outputs in the previous layer"""
		
		cdef float _in, _act, _out, bias;
		cdef str synapse;
		cdef object neuron, weight;
		
		#calculate input & output for every neuron in the layer
		for neuron in self.neurons:
			if type(neuron) != Neuron:
				continue;
				
			_in = 0.0;
			for synapse, weight in neuron.getSynapses().items():
				l, n = parse("L{}N{}", synapse);
				l = int(l);
				n = int(n);
				_in += (outputs[l][n] * weight) if weight != None else outputs[l][n];

			for bias, weight in neuron.getBias().items():
				_in += (bias * weight) if weight != None else bias;

			_in += neuron.getThreshold();
			_act = self.getNetwork().activate(_in);
			_out = self.getNetwork().out(_act);

			neuron.setInput(_in);
			neuron.setValue(_out);

		return [neuron.getValue() for neuron in self.neurons];

	def propagateError(self):
		"""backpropagates the error on the neurons in the layer to the previous layer based on connection"""
		
		cdef object weight;
		cdef str synapse;
		cdef object neuron, targetNeuron;
		
		#set errors on neurons connected to the neurons in the layer
		for neuron in self.neurons:
			for synapse, weight in neuron.getSynapses().items():
				l, n = parse("L{}N{}", synapse);
				l = int(l);
				n = int(n);
				
				#get specific neuron based on L, N
				targetNeuron = self.getNetwork().getLayers()[l].getNeurons()[n];
				if type(targetNeuron) == Neuron:
					if weight == None:
						weight = 1;
					
					targetNeuron.setError(targetNeuron.getError() + (neuron.getError()*weight)*self.getNetwork().derivate(targetNeuron.getInput()));

	def modifyWeights(self):
		"""modifies weights on the neurons in the layer based on error on neurons"""
		
		cdef float bias, learnparam = self.getNetwork().getLearnParam();
		cdef str synapse;
		cdef object neuron, outputNeuron, weight;
		cdef dict newBias, newSynapses;
		
		for neuron in self.neurons:
			newSynapses = {};
			for synapse, weight in neuron.getSynapses().items():
				#constant weight
				if weight == None:
					newSynapses[synapse] = None;
					continue;
				
				l, n = parse("L{}N{}", synapse);
				l = int(l);
				n = int(n);
				outputNeuron = self.getNetwork().getLayers()[l].getNeurons()[n];
				
				newSynapses[synapse] = weight + neuron.getError()*outputNeuron.getValue()*learnparam;

			newBias = {};
			for bias, weight in neuron.getBias().items():
				if weight == None:
					newBias[bias] = weight;
					continue;

				newBias[bias] = weight + neuron.getError()*bias*learnparam;

			neuron.setSynapses(newSynapses);
			neuron.setBias(newBias);

cdef class Constant:
	cdef dict repre;
	
	def __init__(self, dict constRepre):
		"""constRepre: dictionary representing the attributes of a Constant object
		
		creates a constant input with attributes specified in constRepre dictionary"""
		
		self.repre = {};
		self.repre["name"] = constRepre["name"] if "name" in constRepre else "";
		self.repre["input"] = constRepre["input"] if "input" in constRepre else None;

	def getRepre(self):
		"""returns dictionary containing the attributes of an instance"""
		
		return self.repre;
	
	def getCompactRepre(self):
		"""returns dictionary containing the attributes of an instance
		
		default values are omitted"""
		
		cdef dict compact;
		
		compact = {"input":self.repre["input"]};
		if len(self.repre["name"]) > 0:
			compact["name"] = self.repre["name"];
		
		return compact;

	def setName(self, str name):
		"""name: string representing the name of an instance
		
		sets the name of an instance to name"""
		
		self.repre["name"] = name;

	def getName(self):
		"""returns the name representation"""
		
		return self.repre["name"];

	def getValue(self):
		"""returns the constant value of the Constant object"""
		
		return self.repre["input"];

cdef class Variable(Constant):

	def getRepre(self):
		"""returns dictionary containing the attributes of an instance"""
		
		self.repre["input"] = None;
		return self.repre;
	
	def getCompactRepre(self):
		"""returns dictionary containing the attributes of an instance
		
		default values are omitted"""
		
		cdef dict compact;
		
		compact = {"input":None};
		if len(self.repre["name"]) > 0:
			compact["name"] = self.repre["name"];
		
		return compact;
	
	def setValue(self, float value):
		"""value: value to be set on the input
		
		sets the value of the input of an instance"""
		
		value = (round(value*100000))/100000;
		self.repre["input"] = value;

cdef class Neuron:
	cdef object layer;
	cdef dict repre;
	
	def __init__(self, object layer, dict neuronRepre):
		"""layer: instance of a Layer object the neuron is a part of
		neuronRepre: dictionary containing the attributes of a neuron
		
		creates a Neuron object that is a part of layer"""
		
		self.layer = layer;
		
		self.repre = {};
		self.repre["name"] = neuronRepre["name"] if "name" in neuronRepre else "";
		self.repre["synapses"] = neuronRepre["synapses"] if "synapses" in neuronRepre else {};
		self.repre["threshold"] = float(neuronRepre["threshold"]) if "threshold" in neuronRepre else 0.0;
		self.repre["bias"] = neuronRepre["bias"] if "bias" in neuronRepre else {};
		self.repre["input"] = None;
		self.repre["output"] = None;
		self.repre["error"] = 0.0;

	def getRepre(self):
		"""returns dictionary containing the attributes of an instance"""
		
		return self.repre;
	
	def getCompactRepre(self):
		"""returns dictionary containing the attributes of an instance
		
		default values are omitted"""
		
		cdef dict compact, defaults;
		cdef object key, value;
		
		compact = {};
		defaults = {"name":"", "synapses":{}, "threshold":0.0, "bias":{}, "input":None, "output":None, "error":0.0};
		
		for key,value in self.repre.items():
			if value != defaults[key]:
				compact[key] = value;
		
		return compact;

	def getLayer(self):
		"""returns the instance of the object Layer the neuron is a part of"""
		
		return self.layer;

	def setName(self, str name):
		"""sets the 'name' attribute in a neuron representation to name"""
		
		self.repre["name"] = name;

	def getName(self):
		"""returns the name representation of an instance as a string"""
		
		return self.repre["name"];

	def setSynapses(self, dict synapses):
		"""synapses: dictionary representing synapses on the neuron
		
		connects neuron to a neuron in a specific layer with a weight
		{"L0N0":0.75, "L0N1":null}
		null = 1 const"""
		
		cdef str synapse;
		cdef object weight;
		
		for synapse, weight in synapses.items():
			if weight == None:
				continue;
			synapses[synapse] = (round(weight*100000))/100000;
		
		self.repre["synapses"] = synapses;

	def getSynapses(self):
		"""returns dictionary representing the neuron connections"""
		
		return self.repre["synapses"];

	def setThreshold(self, float threshold):
		"""threshold: threshold to be added on the neuron's input during evaluation"""
		
		self.repre["threshold"] = threshold;

	def getThreshold(self):
		"""returns the threshold on the neuron"""
		
		return self.repre["threshold"];

	def setBias(self, dict bias):
		"""bias: dictionary representing the value of a bias and a weight
		
		{1:0.4, 1:null}
		null = 1 const"""
		
		cdef float b;
		cdef object weight;
		
		for b, weight in bias.items():
			if weight == None:
				continue;
			bias[b] = (round(weight*100000))/100000;
		
		self.repre["bias"] = bias;

	def getBias(self):
		"""returns dictionary representing biases connected to the neuron"""
		
		return self.repre["bias"];

	def setInput(self, float value):
		"""value: the value to be set on neuron input"""
		
		self.repre["input"] = (round(value*100000))/100000;

	def getInput(self):
		"""returns the input on the neuron"""
		
		return self.repre["input"];

	def setValue(self, float value):
		"""value: the value to be set on neuron output"""
		
		self.repre["output"] = (round(value*100000))/100000;

	def getValue(self):
		"""returns the value on the neuron output"""
		
		return self.repre["output"];

	def setError(self, error):
		"""error: the value to be set as a neuron's error"""
		
		self.repre["error"] = (round(error*100000))/100000;

	def getError(self):
		"""returns the error on the neuron"""
		
		return self.repre["error"];
		
def baseNetwork():
	"""returns the dict representing the base neural network
	
	1 input, 2 hidden layers (3 & 2 neurons respectivelly), 1 output"""
	
	return {
			"settings":{
				"function":{"name":"sigmoid", "alpha":0.5},
				"learnparam":0.03
			},
			"layers":[
				[
					{"input":None},
					{"input":None}
				],
				[
					{"synapses":{"L0N0":0.5, "L0N1":0.3}},
					{"synapses":{"L0N0":0.3, "L0N1":0.2}},
					{"synapses":{"L0N0":0.8, "L0N1":0.75}}
				],
				[
					{"synapses":{"L1N0":0.1, "L1N1":0.2, "L1N2":0.5}},
					{"synapses":{"L1N0":0.25, "L1N1":0.4, "L1N2":0.7}}
				],
				[
					{"name":"o", "synapses":{"L2N0":0.8, "L2N1":0.6}}
				]
			]
		};

def trainNetwork(object network, list trainingSet, int ratio = 75, int epochs = 5):
	"""network: NeuralNetwork object
	setPath: path to the training set
	epochs: amount of epochs to do (optional, default = 5)
	
	returns a dictionary representing the neural network with modified weights"""
	
	cdef object net;
	cdef int i;
	
	if type(network) == str or type(network) == unicode:
		network = json.loads(network);
	elif type(network) != dict:
		raise Exception("Could not parse Neural Network representation!");
	
	if "settings" not in network:
		raise Exception("Neural Network representation contains no settings!");
	if "layers" not in network:
		raise Exception("Neural Network representation contains no layers!");
	if len(network["layers"]) < 2:
		raise Exception("Neural Network representation has to contain at least 2 layers!");
	if len(trainingSet) < 1:
		raise Exception("Training set is empty!");
	
	net = NeuralNetwork(network);
	
	cdef object counter, reader;
	cdef list inputs, outputs, row, _row, headerRow, _trainingSet;
	cdef float val;
	
	#compare inputs & outputs of the training set to those of a network
	netIO = [len(network["layers"][0]), len(network["layers"][-1])];
	columns = len(trainingSet[0]);
	if columns != netIO[0] + netIO[1]:
		raise Exception("Amount of inputs & outputs in the network doesn't match the dataset. Network IO: {}/{}, Set columns: {}".format(netIO[0], netIO[1], columns));
	
	size = round(len(trainingSet)*(ratio/100.0));
	_testSet = trainingSet[size:];
	trainingSet = trainingSet[0:size];
	
	#row to [input, output] set
	_trainingSet = [];
	for row in trainingSet:
		inputs = row[0:netIO[0]];
		outputs = row[netIO[0]:];
		_row = [inputs, outputs];
			
		_trainingSet.append(_row);
	
	#train for every epoch
	start = time.time();
	for i in range(0, epochs):
		net.train(_trainingSet);
	
	#TODO: test on test set
	
	return {"network":net.getCompactRepre(), "average":(time.time()-start)/epochs, "error":[neuron.getError() for neuron in net.getLayers()[-1].getNeurons()]};