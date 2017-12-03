import math, json, csv, time, itertools, copy;
from parse import parse;

cdef class NeuralNetwork:
	cdef object layers;
	cdef dict settings, functionParams;
	cdef float learnparam;
	
	#NeuralNetwork(networkRepre)
	#
	#networkRepre: dictionary representing the neural network topology
	#Example of network representation can be found in function baseNetwork()
	def __init__(self, dict networkRepre):
		if "settings" not in networkRepre:
			raise Exception("Network representation has to contain network settings")
		if "layers" not in networkRepre:
			raise Exception("Network representation has to contain network layers")
		if len(networkRepre["layers"]) < 2:
			raise Exception("Network representation has to contain at least 2 layers!");
		
		self.settings = networkRepre["settings"];
		if "function" not in self.settings:
			self.settings["function"] = {"name":"sigmoid", "alpha":1.0};
		
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
		
		self.learnparam = self.settings["learnparam"];
		self.layers = [Layer(self, layer) for layer in networkRepre["layers"]];

	#getRepre()
	#
	#Returns the representation of neural network as a dictionary
	def getRepre(self):
		return {"settings":self.settings, "layers":[layer.getRepre() for layer in self.layers]};
	
	#getCompactRepre()
	#
	#Returns the compact representation of neural network as a dictionary
	#Default values are omitted from the representation
	def getCompactRepre(self):
		return {"settings":self.settings, "layers":[layer.getCompactRepre() for layer in self.layers]};

	#getLayers()
	#
	#Returns a list containing Layer objects
	#List represents layers in the network
	def getLayers(self):
		return self.layers;
	
	#getLearnParam()
	#
	#returns a value of learning parameter
	def getLearnParam(self):
		return self.learnparam;
	
	#sigmoid(x)
	#
	#x: value
	#returns the value of sigmoid function for x
	def sigmoid(self, float x):
		cdef float alpha = self.functionParams["alpha"] if "alpha" in self.functionParams else 1.0;
		return 1/(1+math.exp(-x*alpha));
	
	#dsigmoid(x)
	#
	#x: value
	#returns the value of derivated sigmoid function for x
	def dsigmoid(self, float x):
		cdef float alpha = self.functionParams["alpha"] if "alpha" in self.functionParams else 1.0;
		cdef float m = 1+math.exp(-x*alpha);
		return (alpha/m)-(alpha/(m**2));
	
	#dtanh(x)
	#
	#x: value
	#returns the value of derivated tanh function for x
	def dtanh(self, float x):
		return 1-(math.tanh(x)**2);
	
	#activate(x)
	#
	#x: value
	#returns thhe value of activational function for x
	def activate(self, float x):
		return self.functionParams["activate"](x);
	
	#derivate(x)
	#
	#x: value
	#returns the value of derivated activational function for x
	def derivate(self, float x):
		return self.functionParams["derivate"](x);
	
	#out(x)
	#
	#x: value
	#returns the value of output function. Returns x
	def out(self, float x):
		return x;

	#eval([x1, x2])
	#
	#inputs: list representing inputs to the network
	#returns the output of the neural network for a specific input
	def eval(self, list inputs):
		cdef int i;
		cdef object neuron;
		
		#set inputs to variables on input layer
		i = 0;
		for neuron in self.layers[0].getNeurons():
			if type(neuron) == Variable:
				if i >= len(inputs):
					raise Exception("too few inputs");
				
				neuron.setValue(inputs[i]);
				i += 1;
		
		cdef object prevOutputs = [neuron.getValue() for neuron in self.layers[0].getNeurons()];
		cdef int size = len(self.layers);
		
		#propagate inputs forward
		for i in range(1, size):
			prevOutputs = self.layers[i].eval(prevOutputs);

		return prevOutputs;

	#calcErrors([0], [1])
	#
	#result: list representing the output of the network
	#expected: list representing expected output of the network
	#calculates and propagates the error based on the expected value and output of the network
	def calcErrors(self, list result, list expected):
		if len(result) != len(expected):
			raise Exception("length of results is not the same as the expected results");

		cdef object layer;
		cdef object neuron;
		
		for layer in self.layers:
			for neuron in layer.getNeurons():
				if type(neuron) == Neuron:
					neuron.setError(0.0);
		
		cdef int i;
		cdef int size = len(self.layers[-1].getNeurons())
		
		#set errors on output layer
		for i in range(0, size):
			self.layers[-1].getNeurons()[i].setError((expected[i]-result[i])*self.derivate(self.layers[-1].getNeurons()[i].getInput()));

		#propagate errors baclwards
		size = len(self.layers);
		for i in range(1, size):
			self.layers[-i].propagateError();

		#modify weights
		for i in range(1, size):
			self.layers[i].modifyWeights();

	#train(trainingSet)
	#
	#trainingSet: list of rows representing training set, every row contains 2 lists: inputs and outputs of the neural network
	#calculates the errors and adjusts weights based on the training set
	#Example: train([[[1, 1], [1]], [[1, 0], [1]], [[0, 1], [1]], [[0, 0], [0]]])
	def train(self, list trainingSet):
		cdef object row;
		cdef object result;
		
		for row in trainingSet:
			result = self.eval(row[0]);
			self.calcErrors(result, row[1]);
		
cdef class Layer:
	cdef object network;
	cdef object neurons;
	
	#Layer(network, layerRepre)
	#
	#net: instance of the NeuralNetwork
	#layerRepre: list containing dictionaries, where a dictionary is a neuron with specific attributes
	#creates a Layer object in the network with representation specified in layerRepre
	def __init__(self, object net, list layerRepre):
		self.network = net;
		self.neurons = [Neuron(self, repre) if "synapses" in repre and len(repre["synapses"]) > 0 else Variable(repre) if "input" not in repre or repre["input"] == None else Constant(repre) for repre in layerRepre];
	
	#getRepre()
	#
	#analogous function to NeuralNetwork.getRepre()
	def getRepre(self):
		return [neuron.getRepre() for neuron in self.neurons];
	
	#getCompactRepre()
	#
	#analogous function to NeuralNetwork.getCompactRepre()
	def getCompactRepre(self):
		return [neuron.getCompactRepre() for neuron in self.neurons];

	#getNetwork()
	#
	#returns the instance of the network the layer is in
	def getNetwork(self):
		return self.network;

	#getNeurons()
	#
	#returns a list of Neuron objects in the layer
	def getNeurons(self):
		return self.neurons;

	#eval([out1, out2])
	#
	#outputs: list of outputs on the previous layer
	#calculates the outputs of neurons in the layer based on the outputs in the previous layer
	def eval(self, list outputs):
		cdef float _in, _act, _out, bias;
		cdef str synapse;
		cdef object neuron, weight;
			
		for neuron in self.neurons:
			if type(neuron) != Neuron:
				continue;
				
			_in = 0.0;
			for synapse, weight in neuron.getSynapses().items():
				_, n = parse("L{}N{}", synapse);
				n = int(n);
				_in += (outputs[n] * weight) if weight != None else outputs[n];

			for bias, weight in neuron.getBias().items():
				_in += (bias * weight) if weight != None else bias;

			_in += neuron.getThreshold();
			
			_act = self.getNetwork().activate(_in);
			_out = self.getNetwork().out(_act);

			neuron.setInput(_in);
			neuron.setValue(_out);

		return [neuron.getValue() for neuron in self.neurons];

	#propagateError()
	#
	#backpropagates the error on the neurons in the layer
	def propagateError(self):
		cdef object weight;
		cdef str synapse;
		cdef object neuron, targetNeuron;
		
		for neuron in self.neurons:
			for synapse, weight in neuron.getSynapses().items():
				l, n = parse("L{}N{}", synapse);
				l = int(l);
				n = int(n);
				targetNeuron = self.getNetwork().getLayers()[l].getNeurons()[n];
				if type(targetNeuron) == Neuron:
					if weight == None:
						weight = 1;
					
					targetNeuron.setError(targetNeuron.getError() + (neuron.getError()*weight)*self.getNetwork().derivate(targetNeuron.getInput()));

	#modifyWeights()
	#
	#modifies weights on the neurons in the layer based on the errors on neurons
	def modifyWeights(self):
		cdef float bias, learnparam = self.getNetwork().getLearnParam();
		cdef str synapse;
		cdef object neuron, outputNeuron, weight;
		cdef dict newBias, newSynapses;
		
		for neuron in self.neurons:
			newSynapses = {};
			for synapse, weight in neuron.getSynapses().items():
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
	
	#Constant(constRepre)
	#
	#constRepre: dictionary representing the attributes of Constant
	#creates a constant input with attributes specified in constRepre dictionary
	def __init__(self, dict constRepre):
		self.repre = {};
		self.repre["name"] = constRepre["name"] if "name" in constRepre else "";
		self.repre["input"] = constRepre["input"] if "input" in constRepre else None;

	#getRepre()
	#
	#analogous to NeuralNetwork.getRepre()
	def getRepre(self):
		return self.repre;
	
	#getCompactRepre()
	#
	#analogous to NeuralNetwork.getCompactRepre()
	def getCompactRepre(self):
		cdef dict compact;
		
		compact = {"input":self.repre["input"]};
		if len(self.repre["name"]) > 0:
			compact["name"] = self.repre["name"];
		
		return compact;

	#setName(name)
	#
	#name: string representing the name of a neuron
	#sets the name of a neuron to a string
	def setName(self, str name):
		self.repre["name"] = name;

	#getName()
	#
	#returns the name of a neuron
	def getName(self):
		return self.repre["name"];

	#getValue()
	#
	#returns the constant value of the Constant object
	def getValue(self):
		return self.repre["input"];

#Constant subclass
cdef class Variable(Constant):

	#getRepre()
	#
	#analogous to Constant.getRepre()
	#returns input attribute as a NoneType object
	def getRepre(self):
		self.repre["input"] = None;
		return self.repre;
	
	#getCompactRepre()
	#
	#analogous to Constant.getCompactRepre()
	def getCompactRepre(self):
		cdef dict compact;
		
		compact = {"input":None};
		if len(self.repre["name"]) > 0:
			compact["name"] = self.repre["name"];
		
		return compact;
	
	#setValue(value)
	#
	#value: value to be set on the input
	#sets the value on the input of variable
	def setValue(self, float value):
		value = (round(value*100000))/100000;
		self.repre["input"] = value;

cdef class Neuron:
	cdef object layer;
	cdef dict repre;
	
	#Neuron(layer, neuronRepre)
	#
	#layer: instance of a Layer the neuron is part of
	#neuronRepre: dictionary containing the attributes of a neuron
	#creates a Neuron object that is a part of layer
	def __init__(self, object layer, dict neuronRepre):
		self.layer = layer;
		
		self.repre = {};
		self.repre["name"] = neuronRepre["name"] if "name" in neuronRepre else "";
		self.repre["synapses"] = neuronRepre["synapses"] if "synapses" in neuronRepre else {};
		self.repre["threshold"] = float(neuronRepre["threshold"]) if "threshold" in neuronRepre else 0.0;
		self.repre["bias"] = neuronRepre["bias"] if "bias" in neuronRepre else {};
		self.repre["input"] = None;
		self.repre["output"] = None;
		self.repre["error"] = 0.0;

	#getRepre()
	#
	#analogous to Variable.getRepre()
	def getRepre(self):
		return self.repre;
	
	#getCompactRepre()
	#
	#analogous to Variable.getCompactRepre()
	def getCompactRepre(self):
		cdef dict compact, defaults;
		cdef object key, value;
		
		compact = {};
		defaults = {"name":"", "synapses":{}, "threshold":0.0, "bias":{}, "input":None, "output":None, "error":0.0};
		
		for key,value in self.repre.items():
			if value != defaults[key]:
				compact[key] = value;
		
		return compact;

	#getLayer()
	#
	#returns the instance of the object Layer the neuron is a part of
	def getLayer(self):
		return self.layer;

	#setName(name)
	#
	#analogous to Variable.setName()
	def setName(self, str name):
		self.repre["name"] = name;

	#getName()
	#
	#analogous to Variable.getName()
	def getName(self):
		return self.repre["name"];

	#setSynapses(synapses)
	#
	#synapses: dictionary representing synapses on the neuron
	#connects neuron to a neuron in a specific layer with a weight
	def setSynapses(self, dict synapses):
		cdef str synapse;
		cdef object weight;
		
		for synapse, weight in synapses.items():
			if weight == None:
				continue;
			synapses[synapse] = (round(weight*100000))/100000;
		
		self.repre["synapses"] = synapses;

	#getSynapses()
	#
	#returns dictionary representing neuron connections
	def getSynapses(self):
		return self.repre["synapses"];

	#setThreshold(threshold)
	#
	#threshold: threshold to be added on the neuron's input
	def setThreshold(self, float threshold):
		self.repre["threshold"] = threshold;

	#getThreshold()
	#
	#returns the threshold on the neuron
	def getThreshold(self):
		return self.repre["threshold"];

	#setBias(bias)
	#
	#bias: dictionary representing the value of a bias and a weight
	def setBias(self, dict bias):
		cdef float b;
		cdef object weight;
		
		for b, weight in bias.items():
			if weight == None:
				continue;
			bias[b] = (round(weight*100000))/100000;
		
		self.repre["bias"] = bias;

	#getBias()
	#
	#returns dictionary representing biases connected to the neuron
	def getBias(self):
		return self.repre["bias"];

	#setInput(value)
	#
	#value: the value to be set on neuron input
	def setInput(self, float value):
		self.repre["input"] = (round(value*100000))/100000;

	#getInput()
	#
	#returns the input on the neuron
	def getInput(self):
		return self.repre["input"];

	#setValue(value)
	#
	#value: the value to be set on neuron output
	def setValue(self, float value):
		self.repre["output"] = (round(value*100000))/100000;

	#getValue()
	#
	#returns the value on the neuron output
	def getValue(self):
		return self.repre["output"];

	#setError(error)
	#
	#error: the value to be set as a neuron's error
	def setError(self, error):
		self.repre["error"] = (round(error*100000))/100000;

	#getError()
	#
	#returns the error on the neuron
	def getError(self):
		return self.repre["error"];
		
#baseNetwork()
#
#returns the dict representing the base neural network
#1 input, 2 hidden layers (3 & 2 neurons respectivelly), 1 output
def baseNetwork():
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

#trainNetwork(network, setPath, epochs = 5)
#
#network: NeuralNetwork object
#setPath: path to the training set
#epochs: amount of epochs to do (optional, default = 5)
#returns a dictionary representing the neural network with modified weights
def trainNetwork(object network, str setPath, int epochs = 5):
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
	
	net = NeuralNetwork(network);
	
	cdef object counter, reader;
	cdef list inputs, outputs, row, _row, headerRow, _trainingSet;
	cdef float val;
	with open(setPath, newline="") as csvfile:
		counter, reader = itertools.tee(csv.reader(csvfile, delimiter=","));
		
		headerRow = next(counter);
		columns = len(headerRow);
		del counter;
		
		netIO = [len(network["layers"][0]), len(network["layers"][-1])];
		if columns != netIO[0] + netIO[1]:
			columnIO = [0, 0];
			for column in headerRow:
				if column.find("x") > -1:
					columnIO[0] += 1;
			
			columnIO[1] = columns - columnIO[0];
			raise Exception("Amount of inputs & outputs in the network doesn't match the dataset. Network IO: {}/{}, Set IO: {}/{}".format(netIO[0], netIO[1], columnIO[0], columnIO[1]));
		
		_trainingSet = [];
		for row in reader:
			inputs = [float(row[i]) for i in range(0, netIO[0])];
			outputs = [float(row[i]) for i in range(netIO[0], netIO[1]+netIO[0])];
			_row = [inputs, outputs];
			
			_trainingSet.append(_row);
	
	start = time.time();
	for i in range(0, epochs):
		print("{}/{}".format(i, epochs));
		net.train(_trainingSet);
	
	return {"network":net.getCompactRepre(), "average":(time.time()-start)/epochs, "error":[neuron.getError() for neuron in net.getLayers()[-1].getNeurons()]};