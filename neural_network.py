import glob
import sys
import os
import os.path
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


class NeuralNetwork:
		
	def __init__(self, c, w, stride, epochs, learning_rate, k, directory_path, X_train, Y_train, X_test, Y_test,
				row_train, column_train, row_test, column_test, unique_targets_list):
		
		#Inititialise weights in the range [-0.5, 0.5] between all layers 
		self.input_to_hl1_weights =  np.random.uniform(-0.5,0.5,(c, w))
		self.hl1_to_hl2_weights =  np.random.uniform(-0.5,0.5,(c, c))
		self.hl2_to_output_weights = np.random.uniform(-0.5,0.5,(k, c))	
		
		self.unique_targets_list = unique_targets_list
		
		self.net_h1 = []
		self.net_h2 = []
		self.net_o = []
		
		self.out_h1 = []
		self.out_h2 = []
		self.out_o = []
		
		#Inititialise Delta weights 
		self.input_to_hl1_weights_delta =  np.zeros((c, w))
		self.hl1_to_hl2_weights_delta =  np.zeros((c, c))
		self.hl2_to_output_weights_delta =np.zeros((k, c))
				
		#I/O Files
		dataset_name = os.path.basename(directory_path)
		print("Dataset Name: ", dataset_name)
		
		log_file_name = dataset_name + ".log"
		print("Log file name: ", log_file_name)
		if os.path.isfile(log_file_name):
			os.remove(log_file_name)
		self.f_log = open(log_file_name, "a")
		
		dataset_final_weights_file_name = dataset_name + "_FinalWeights.txt"
		print("Final Weights file name: ",dataset_final_weights_file_name)
		if os.path.isfile(dataset_final_weights_file_name):
			os.remove(dataset_final_weights_file_name)
		self.f_dataset_final_weights = open(dataset_final_weights_file_name, "a")
		
		dataset_window_prediction = dataset_name + "_WindowPrediction.txt"
		print("Window Prediction file name: ",dataset_window_prediction)
		if os.path.isfile(dataset_window_prediction):
			os.remove(dataset_window_prediction)
		self.f_window_prediction = open(dataset_window_prediction, "a")
		

		print('c = %d'%c, file=self.f_log)
		print('w = %d'%w, file=self.f_log)
		print('stride = %d'%stride, file=self.f_log)
		print('epochs = %d'%epochs, file=self.f_log)
		print('k = %d'%k, file=self.f_log)
		
		#Train
		self.train(c, w, stride, epochs, learning_rate, k, X_train, Y_train, row_train, column_train)
		
		#Test
		self.test(c, w, stride, k, X_test, Y_test, row_test, column_test)
		
		#Closing files
		self.f_log.close()
		self.f_dataset_final_weights.close()
		self.f_window_prediction.close()
		
		
	def init_delta_weights(self, c, w, k):
		self.input_to_hl1_weights_delta =  np.zeros((c, w))
		self.hl1_to_hl2_weights_delta =  np.zeros((c, c))
		self.hl2_to_output_weights_delta =np.zeros((k, c))
		
	def activation_functions(self, X, activation_function_name):
		
		Z = None
			
		if activation_function_name.lower()  == "leaky_relu":
			X[ X < 0 ] *= 0.01
			Z = X
		elif activation_function_name.lower()  == "sigmoid":
			Z = 1.0/(1.0+np.exp(-np.double(X)))
		elif activation_function_name.lower()  == "softmax":
			e_x = np.exp(X - np.max(X))
			Z = e_x / e_x.sum(axis=0)
		else:
			Z = None
			
		return Z
		
		
	def activation_functions_derivatives(self, X, activation_function_name):
	
		Z = None
		
		if activation_function_name.lower()  == "leaky_relu":
			Z = np.clip(X > 0, 0.01, 1.0)
		elif activation_function_name.lower()  == "sigmoid":
			Z = self.activation_functions(X, "sigmoid") *(1- self.activation_functions(X, "sigmoid"))
		else:
			Z = None
			
		return Z
		
		
	def cross_entropy_loss(self, actual, predicted, epsilon=1e-12):
	
		predicted = np.clip(predicted, epsilon, 1. - epsilon)
		N = predicted.shape[0]
		ce = -np.sum(actual*np.log(predicted+1e-9))/N
		
		return ce
	
	
	def most_common(self, lst):
		return max(set(lst), key=lst.count)
	
	
	def forward_propagation(self, cbywVector):
	
		#Input layer to Hidden Layer 1
		self.net_h1 = np.multiply(cbywVector, self.input_to_hl1_weights).sum(1)
		self.out_h1 = self.activation_functions(self.net_h1, "leaky_relu")

		#Hidden Layer 1 to Hidden Layer 2
		self.net_h2 = np.multiply(self.out_h1, self.hl1_to_hl2_weights).sum(1)
		self.out_h2 = self.activation_functions(self.net_h2, "sigmoid")

		#Hidden Layer 2 to Output Layer		
		self.net_o = np.multiply(self.out_h2 , self.hl2_to_output_weights).sum(1)
		self.out_o = self.activation_functions(self.net_o, "softmax")	
		
		
	def back_propagation(self, cBywVector, c, w, k, learning_rate, Y):
		
		#Inititialise partial derivative
		partial_derivative = [None]*3
		
		#Output - HL2 Layer
		partial_derivative[2] = self.out_o - Y
		
		#HL2 - HL1 Layer
		partial_derivative[1] = [0.0] * c
		for i in range(0, c):
			for j in range(0, k):
				partial_derivative[1][i] += partial_derivative[2][j] * self.hl2_to_output_weights[j][i]
			partial_derivative[1][i] *= self.activation_functions_derivatives(self.out_h2[i], "sigmoid")
			
		#HL1 - Input Layer
		partial_derivative[0] = [0.0] * c
		for i in range(0, c):
			for j in range(0, c):
				partial_derivative[0][i] += partial_derivative[1][j] * self.hl1_to_hl2_weights[j][i]
			partial_derivative[0][i] *= self.activation_functions_derivatives(self.out_h1[i], "leaky_relu")

		
		#Update Weights within an epoch
		for i in range(0, k):
			temp = partial_derivative[2][i] * learning_rate
			for j in range(0, c):
				self.hl2_to_output_weights_delta[i][j] -= temp * self.out_h2[j]

		for i in range(0, c):
			temp = partial_derivative[1][i] * learning_rate* 0.8
			for j in range(0, c):
				self.hl1_to_hl2_weights_delta[i][j] -= temp * self.out_h1[j]
		
		for i in range(0, c):
			temp = partial_derivative[0][i] * learning_rate
			for j in range(0, w):
				self.input_to_hl1_weights_delta[i][j] -= temp * cBywVector[i][j]
				
		
	def learning_rate_decay(self, learning_rate, e):
	
		lr_list_decay = [0.6, 0.3, 0.15, 0.08, 0.01]
		step_count = 0
		if(e % 150 == 0):
			learning_rate = lr_list_decay[step_count] * learning_rate
			print('\nLearning rate: ',learning_rate)
			step_count += 1
			if step_count >= 4:
				step_count = 4
				
		return learning_rate
	
	
	def train(self, c, w, stride, epochs, learning_rate, k, X_train, Y_train, row_train, column_train):
		
		common_correct_preds_num = 0
		
		print("\n--------Training--------", file=self.f_log)
		print("\n--------Training--------", file=self.f_window_prediction)
		print("\n--------Training--------")
		for e in range(0, epochs):
			actual_vals_per_epoch = []
			predicted_vals_per_epoch = []
			loss = 0.0
			learning_rate = self.learning_rate_decay(learning_rate, e)
			self.counter = 0
			for rr in range(0, row_train):
				one_hot_encoded =  [1 if Y_train[rr] == unique else 0 for unique in self.unique_targets_list]
				most_common_prediction_row = []
				if(e == epochs - 1):
						print('\nFor Row[%d]:' %(rr), file=self.f_window_prediction)
				for ww in range(0, column_train - w, stride):
				
					#Get window
					cBywVector = []
					cBywVector = [X_train[cc][rr:rr+1,ww :ww + w] for cc in range(0, c)]
					cBywVector = np.array(cBywVector)
					cBywVector = cBywVector[:, 0, :]
										
					#Forward Propagation	
					self.forward_propagation(cBywVector)
					
					#Check predicted value
					index_max = np.argmax(self.out_o)
					predicted_value = self.unique_targets_list[index_max]
					most_common_prediction_row.append(predicted_value)
					
					#Window Prediction for last epoch.
					if(e == epochs - 1):
						print('Window Prediction: %d, Expected: %d' %(predicted_value, Y_train[rr]), file=self.f_window_prediction)
					
					#Check Loss/Cost
					loss +=  self.cross_entropy_loss(one_hot_encoded, self.out_o)
					
					#Back Propagation
					self.back_propagation(cBywVector, c, w, k, learning_rate, one_hot_encoded)
					
					self.counter += 1
					
				#Voting for most common prediction
				most_common_pred = self.most_common(most_common_prediction_row)
	
				#actual_vals_per_epoch.append(Y_train[rr])
				predicted_vals_per_epoch.append(most_common_pred)
				
				if(e == epochs - 1):
					print('Most Common Prediction: %d, Expected: %d' %(most_common_pred, Y_train[rr]), file=self.f_log)
					if(most_common_pred == Y_train[rr]):
						common_correct_preds_num += 1				
			
			
			#Update Weights
			self.hl2_to_output_weights += self.hl2_to_output_weights_delta/self.counter
			self.hl1_to_hl2_weights += self.hl1_to_hl2_weights_delta/self.counter
			self.input_to_hl1_weights += self.input_to_hl1_weights_delta/self.counter
			
			
			#Clear delta weights with zeroes.
			self.init_delta_weights(c, w, k)
				
			#Calculating Loss and Accuracy per epoch	
			print('\nEpoch %d/%d'% (e + 1, epochs))
			accuracy_per_epoch = accuracy_score(Y_train, predicted_vals_per_epoch)
			print('Loss: ', np.around(loss/self.counter, decimals=2))
			print('Accuracy: ', np.around(accuracy_per_epoch, decimals=4), '\n')

			if(e == epochs - 1):
				print("\n-----------Confusion Matrix for train-----------", file=self.f_log)
				conf_matrix_train = confusion_matrix(Y_train, predicted_vals_per_epoch)
				print(conf_matrix_train, file=self.f_log)
				
				print("\n-------Accuracy Metrics for train-------", file=self.f_log)
				print("\n-------Accuracy Metrics for train-------")
				accuracy_most_common = common_correct_preds_num/row_train
				print("Most Common Accuracy via voting strategy: ", (np.around(accuracy_most_common, decimals=4)), file=self.f_log)
				print("Most Common Accuracy via voting strategy: ", (np.around(accuracy_most_common, decimals=4)))
			

			
		self.hl1_to_hl2_weights =  2.*(self.hl1_to_hl2_weights- np.min(self.hl1_to_hl2_weights))/np.ptp(self.hl1_to_hl2_weights)-1
		print("Weights between Input layer and Hidden Layer 1\n" , self.input_to_hl1_weights, file=self.f_dataset_final_weights)
		print("\nWeights between Hidden Layer 1 and Hidden Layer 2\n" , self.hl1_to_hl2_weights, file=self.f_dataset_final_weights)
		print("\nWeights between Hidden Layer 2 and Output Layer\n" ,self.hl2_to_output_weights, file=self.f_dataset_final_weights)
		
	
	def test(self, c, w, stride, k, X_test, Y_test, row_test, column_test):
		
		common_correct_preds_counter = 0
		
		actual_vals_per_row = []
		predicted_vals_per_row = []
		
		actual_vals_per_window = []
		predicted_vals_per_window = []
		
		print("\n--------Testing--------")
		print("\n--------Testing--------", file=self.f_log)
		print("\n--------Testing--------", file=self.f_window_prediction)
		for rr in range(0, row_test): 
			predicted_window = []
			print('\nFor Row[%d]:' %(rr), file=self.f_window_prediction)
			for ww in range(0, column_test - w, stride):
				
				#Get window
				cBywVector = []
				cBywVector = [X_test[cc][rr:rr+1,ww :ww + w] for cc in range(0, c)]
				cBywVector = np.array(cBywVector)
				cBywVector = cBywVector[:, 0, :]
					
				#Forward Propagation
				self.forward_propagation(cBywVector)
				
				#Check predicted value
				index_max = np.argmax(self.out_o)
				predicted_value = self.unique_targets_list[index_max]
				predicted_window.append(predicted_value)
				
				#Window prediction
				print('Window Prediction: %d, Expected: %d' %(predicted_value, Y_test[rr]), file=self.f_window_prediction)
				
				actual_vals_per_window.append(Y_test[rr])
				predicted_vals_per_window.append(predicted_value)
									
			#Calculating Accuracy for Most Common prediction
			most_common_pred = self.most_common(predicted_window)
			if(most_common_pred == Y_test[rr]):
				common_correct_preds_counter += 1
				
			print('Most Common Prediction: %d, Expected: %d' %(most_common_pred, Y_test[rr]), file=self.f_log)
			predicted_vals_per_row.append(most_common_pred)
			sys.stdout.write(".")
			sys.stdout.flush()
			
		print("\n-----------Confusion Matrix for test-----------", file=self.f_log)
		conf_matrix_test = confusion_matrix(Y_test, predicted_vals_per_row)
		print(conf_matrix_test, file=self.f_log)
	
	
		print("\n-------Accuracy Metrics for test-------", file=self.f_log)
		print("\n-------Accuracy Metrics for test-------")
		
		accuracy_data_chunks = accuracy_score(actual_vals_per_window, predicted_vals_per_window)
		print("Accuracy on data chunks: ", (np.around(accuracy_data_chunks, decimals=4)), file=self.f_log)
		print("Accuracy on data chunks: ", (np.around(accuracy_data_chunks, decimals=4)))
		
		accuracy_most_common = common_correct_preds_counter/row_test
		print("Most Common Accuracy via voting strategy: ", np.around(accuracy_most_common, decimals=4), file=self.f_log)
		print("Most Common Accuracy via voting strategy: ", np.around(accuracy_most_common, decimals=4))

		
		
		
'''list subdirectories in a directory path'''
def folders_in(path_to_parent):
	for fname in os.listdir(path_to_parent):
		if os.path.isdir(os.path.join(path_to_parent,fname)):
			yield os.path.join(path_to_parent,fname)
						

'''Read arff file format'''
def read_arff(w, stride, epochs, learning_rate, dir_path):
	
	c = 0
	k = 0
	
	#Names of the files
	training_files = []
	testing_files = []

	#Dataframe of training and testing files
	df_train = []
	df_test = []

	#Data
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	unique_targets_list = []

	#total
	row_train, row_test, column_train, column_test = 0, 0, 0, 0
	
	
	print("ARFF file format found!")
	for file in glob.glob(os.path.join(dir_path, '*_TRAIN.arff')):
		c += 1
		training_files.append(file)
	

	for file in glob.glob(os.path.join(dir_path, '*_TEST.arff')):
		testing_files.append(file)


	print("c = ", c)
	print("w = ", w)
	print("stride = ", stride)
	print("epochs = ", epochs)
	print("Learning rate = ", learning_rate)
	print("Given Directory Path: ",dir_path)
	
	for i in range(0, c):
		data_train = arff.loadarff(training_files[i])   
		df_train.insert(i,pd.DataFrame(data_train[0]))
		X_train.insert(i, df_train[i].rename_axis('ID').values)  #target column still remains
	
		data_test = arff.loadarff(testing_files[i]) 
		df_test.insert(i, pd.DataFrame(data_test[0]))
		X_test.insert(i, df_test[i].rename_axis('ID').values)    #target column still remains
	
	row_train, column_train = df_train[0].shape
	row_test, column_test = df_test[0].shape
	print("column_train" , column_train)

	if(w > column_train - 1):
		w = column_train - 1
	
	Y_train = X_train[0][:,column_train - 1]
	Y_train = list(map(float, Y_train))
	Y_test = X_test[0][:, column_test - 1] 
	Y_test = list(map(float, Y_test))
	X_train = np.array(X_train)
	X_train = X_train[:,:,:column_train -1]
	X_test = np.array(X_test)
	X_test= X_test[:,:,:column_test -1]
	unique_targets_list = np.unique(Y_train)
	k = len(unique_targets_list)
	print('k = %d'%k)

	#Calling NeuralNetwork class
	nn = NeuralNetwork(c, w, stride, epochs, learning_rate, k, dir_path, X_train, Y_train, X_test, Y_test,
				row_train, column_train, row_test, column_test, unique_targets_list)
	
	
'''Read csv file format'''				
def read_csv(w, stride, epochs, learning_rate, dir_path):

	c = 0
	k = 0
	
	#Names of the files
	training_files = []
	testing_files = []

	#Dataframe of training and testing files
	df_train = []
	df_test = []

	#Data
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	unique_targets_list = [] #Unique

	#total
	row_train, row_test, column_train, column_test = 0, 0, 0, 0
	print("CSV file format found!")
			
	for file in glob.glob(os.path.join(dir_path, '*_TRAIN.csv')):
		c += 1
		training_files.append(file)
	
	for file in glob.glob(os.path.join(dir_path, '*_TEST.csv')):
		testing_files.append(file)
		
		
	print("c = ", c)
	print("w = ", w)
	print("stride = ", stride)
	print("epochs = ", epochs)
	print("Learning rate = ", learning_rate)
	print("Given Directory Path: ",dir_path)

	for i in range(0, c):
		data_train = pd.read_csv(training_files[i], sep=';', encoding='utf-8', quotechar='"', decimal=',')  
		df_train.insert(i,data_train.values)
		X_train.insert(i, df_train[i])
	
		data_test = pd.read_csv(testing_files[i], sep=';', encoding='utf-8', quotechar='"', decimal=',')  
		df_test.insert(i,data_test.values)
		X_test.insert(i, df_test[i])
	
	row_train, column_train = df_train[0].shape
	row_test, column_test = df_test[0].shape
	print("column_train" , column_train)
	
	if(w > column_train - 1):
		w = column_train - 1
	
	Y_train = X_train[0][:,0]
	Y_train = list(map(float, Y_train))
	Y_test = X_test[0][:,0]
	Y_test = list(map(float, Y_test))
	X_train = np.array(X_train)
	X_train = X_train[:,:,1:]
	X_test = np.array(X_test)
	X_test= X_test[:,:,1:]
	unique_targets_list = np.unique(Y_train)
	k = len(unique_targets_list)
	print('k = %d'%k)
	
	#Calling NeuralNetwork class
	nn = NeuralNetwork(c, w, stride, epochs, learning_rate, k, dir_path, X_train, Y_train, X_test, Y_test,
				row_train, column_train, row_test, column_test, unique_targets_list)
	
	
'''Main Function'''
def main():

	#Command Line Arguments
	w, stride, epochs, learning_rate, dir_path = 0, 0, 0, 0.0,  ""
	if len(sys.argv) == 6:
		dir_path = sys.argv[1]
		w = int(sys.argv[2])
		stride = int(sys.argv[3])
		epochs = int(sys.argv[4])
		learning_rate = float(sys.argv[5])
		
	else:
		print("Incorrect number of arguments")
		exit()
	
	#List of subdirectories 
	subdirectories_list = []
	
	#Check if there are subdirectories in the given path
	if(folders_in(dir_path)):
		subdirectories_list = list(folders_in(dir_path))
		
		#No subdirectories
		if(len(subdirectories_list) == 0): 
			if(glob.glob(os.path.join(dir_path, '*.arff'))): #ARFF
				read_arff(w, stride, epochs, learning_rate,dir_path)
			
			elif(glob.glob(os.path.join(dir_path, '*.csv'))): #CSV
				read_csv(w, stride, epochs, learning_rate,dir_path)
			
			else:
				print("Invalid file type")
		
		else:
			print("Given subdirectories:\n", subdirectories_list)
			print()
			for sub_dir in subdirectories_list:
				if(glob.glob(os.path.join(sub_dir, '*.arff'))): #ARFF
					print("INIT: " , sub_dir)
					print()
					read_arff(w, stride, epochs, learning_rate,sub_dir)
					print('\n---------------------\n')
					
				elif(glob.glob(os.path.join(sub_dir, '*.csv'))): #CSV
					print("INIT: " , sub_dir)
					print()
					read_csv(w, stride, epochs, learning_rate,sub_dir)
					print('\n---------------------\n')
					
				else: #Incorrect file format
					print("INIT: " , sub_dir)
					print()
					print("Invalid file type")
					print('\n---------------------\n')

'''Calling main function'''
if __name__== "__main__":
	main()