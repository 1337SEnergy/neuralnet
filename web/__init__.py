"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)

from os import listdir
from os.path import isfile, join
import csv, random;
def loadDataSet(setName):
	with open("./dataSets/{}".format(setName), newline="") as csvfile:
		reader = csv.reader(csvfile, delimiter=",");
		
		#try to add first row, fails if its a header
		trainingSet = [];
		headerRow = next(reader);
		try:
			_row = [float(r) for r in headerRow];
			trainingSet.append(_row);
		except:
			pass;
		
		#append rows to training set
		for row in reader:
			_row = [float(r) for r in row];
			trainingSet.append(_row);
	
	#shuffle set
	random.shuffle(trainingSet);
	return trainingSet;

files = [f for f in listdir("./dataSets") if isfile(join("./dataSets", f))]
dataSets = {setName:loadDataSet(setName) for setName in files};

import web.views