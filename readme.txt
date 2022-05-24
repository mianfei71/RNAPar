1. Environment 
* Ubuntu 18.04.5 LTS
* TensorFlow 1.14.0
* Keras 2.2.4
* Cuda V9.1.85
* Cudnn 7.6
* H5py 2.10.0

2. Input
* Fasta file with one or more RNA sequences. Only A, C, G, U, and ‘-’ are allowed in RNA sequence.
* Then length of RNA sequence should be 200 (including ‘-’ at the end of the sequences).

3. Output
* 1. The output of RNA-par will be put into file "./predict/test.data"
* 2. The Number of line is equal to the number of sequences in input.

4 Command
* 1. A command example: python predict.py -i ./data/test.fasta -o ./predict/test.data -w ./models/weight-1.h5 -K 6 -C 61 -U 115 -N 53
* 2. RNA-par will create a model with hyperparameters (K, C, U, and N) and parameters in file weight-1.h5, then predict the labels for the sequences in test.fasta with model.h5, and output the predictions to ./result/test.data.
