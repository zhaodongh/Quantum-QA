# Quantum-QA

This is the code to reproduce the model in [End-to-End Quantum-like Language Models](http://oro.open.ac.uk/52878/1/16720-72069-3-SM.pdf)

##Environment
	python 3.5
	theano

## Setup
	$ git clone https://github.com/zhaodongh/Quantum-QA.git

## Unzip Data

Unzip the data.rar,and then the folder should look like this:

	$ cd data/
	$ ls
	embedding_dir  statc_fatures  trec  trec-all  wiki
	$ cd ..

## To run the code
	$ python config.py
	$ python parse.py
	$ python main.py


