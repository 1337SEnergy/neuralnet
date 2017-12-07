from datetime import datetime
from os import listdir
from os.path import isfile, join
from flask import render_template, request, jsonify, json, redirect, url_for;
from web import app, dataSets, loadDataSet;
from neuralNet import trainNetwork, NeuralNetwork, baseNetwork;
from werkzeug.utils import secure_filename
import ast;

def compactJson(_json):
	toFormat = {"\t\t\t{\n\t\t\t\t":"\t\t\t{", "\n\t\t\t}":"}", "\n\t\t\t\t\t":"", "\n\t\t\t\t}":"}", "\n\t\t\t\t":""};
	_str = json.dumps(_json, sort_keys = False, indent = "\t");
	
	for key,value in toFormat.items():
		_str = _str.replace(key, value);
	
	return _str;

def allowedFile(fileName):
	return '.' in fileName and fileName.rsplit('.', 1)[1].lower() == "csv";

@app.route('/')
def home():
	network = request.args["network"] if "network" in request.args else baseNetwork();
	if type(network) == str:
		network = network.replace("'", "\"");
		network = ast.literal_eval(network);
	
	network = compactJson(network);
	files = [f for f in listdir("./dataSets") if isfile(join("./dataSets", f))]
	
	return render_template(
		"index.html",
		network=network,
		files = files,
		year=datetime.now().year,
	)

@app.route('/upload', methods=["GET", "POST"])
def upload_file():
	network = baseNetwork();
	if "network" in request.args:
		_str = request.args["network"];
		_str = _str.replace("\\t", "");
		_str = _str.replace("\t", "");
		_str = _str.replace("\\n", "");
		_str = _str.replace("\n", "");
		_str = _str.replace("null", "None");
		network = ast.literal_eval(_str);
	
	if request.method == "POST":
		if "file" not in request.files:
			return redirect(url_for(".home", network=network));

		file = request.files["file"];
		if file.filename == "":
			return redirect(url_for(".home", network=network));

		if file and allowedFile(file.filename):
			filename = secure_filename(file.filename)
			try:
				file.save(join("D:/home/site/wwwroot/dataSets", filename));
			except:
				file.save(join("dataSets", filename));
	
	return redirect(url_for(".home", network=network));

@app.route("/train")
def train():
	'''
	try:
		if "trainingSet" not in request.args:
			raise Exception("No training set specified");
			
		if request.args["trainingSet"] not in dataSets:
			if request.args["trainingSet"] not in listdir("./dataSets"):
				raise Exception("Yo how you got that data set?");
			
			dataSets[request.args["trainingSet"]] = loadDataSet(request.args["trainingSet"]);

		network = baseNetwork();
		if "network" in request.args:
			_str = request.args["network"];
			_str = _str.replace("\\t", "");
			_str = _str.replace("\t", "");
			_str = _str.replace("\\n", "");
			_str = _str.replace("\n", "");
			_str = _str.replace("null", "None");
			network = ast.literal_eval(_str);
		
		ratio = int(request.args["train-ratio"]) if "train-ratio" in request.args else 75;
		if ratio > 100 or ratio < 0:
			ratio = 75;
		
		epochs = int(request.args["epochs"]) if "epochs" in request.args else 5;
		epochs = min(epochs, 50);

		result = trainNetwork(network, dataSets[request.args["trainingSet"]], ratio, epochs);
		return jsonify({"success":True, "network":compactJson(result["network"]), "message":"Average time per epoch: {} sec".format(result["average"]), "errorVal":result["error"]});

	except Exception as e:
		return jsonify({"success":False, "error":str(e)});
	'''
	
	if "trainingSet" not in request.args:
		raise Exception("No training set specified");
			
	if request.args["trainingSet"] not in dataSets:
		if request.args["trainingSet"] not in listdir("./dataSets"):
			raise Exception("Yo how you got that data set?");
			
		dataSets[request.args["trainingSet"]] = loadDataSet(request.args["trainingSet"]);

	network = baseNetwork();
	if "network" in request.args:
		_str = request.args["network"];
		_str = _str.replace("\\t", "");
		_str = _str.replace("\t", "");
		_str = _str.replace("\\n", "");
		_str = _str.replace("\n", "");
		_str = _str.replace("null", "None");
		network = ast.literal_eval(_str);
		
	ratio = int(request.args["train-ratio"]) if "train-ratio" in request.args else 75;
	if ratio > 100 or ratio < 0:
		ratio = 75;
		
	epochs = int(request.args["epochs"]) if "epochs" in request.args else 5;
	epochs = max(epochs, 0);
	epochs = min(epochs, 5);

	result = trainNetwork(network, dataSets[request.args["trainingSet"]], ratio, epochs);
	return jsonify({"success":True, "network":compactJson(result["network"]), "message":"Average time per epoch: {} sec".format(result["average"]), "epochErrors":result["epochErrors"]});


@app.route("/run")
def run():
	try:
		if "network" not in request.args:
			raise Exception("undefined Neural Network");
		if "inputs" not in request.args:
			raise Exception("undefined inputs");

		_str = request.args["network"];
		_str = _str.replace("\\t", "");
		_str = _str.replace("\t", "");
		_str = _str.replace("\\n", "");
		_str = _str.replace("\n", "");
		_str = _str.replace("null", "None");
		network = ast.literal_eval(_str);
		net = NeuralNetwork(network);

		return jsonify({"success":True, "message":str(net.eval(ast.literal_eval(request.args["inputs"])))});
	except Exception as e:
		return jsonify({"success":False, "error":str(e)});