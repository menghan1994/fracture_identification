import numpy as np
import pandas as pd
import matplotlib.image as mpimg

class DataLoader_for_training():
    def __init__(self, original_picture, labled_picture, sample_size_x = 5,sample_size_y = 5,data_type = 'uint16'):

        self.original_data = mpimg.imread(original_picture)
        self.labled_data = mpimg.imread(labled_picture)
        self.size_X = len(self.original_data[0,:])
        self.size_Y = len(self.original_data[:,0])
        self.sample_size_x = sample_size_x
        self.sample_size_y = sample_size_y
        self.data_type = data_type
        
    def generate_training_data(self):      
        length_x = self.size_X - 2*self.sample_size_x -1
        length_y = self.size_Y - 2*self.sample_size_y -1                
        train_x = np.zeros((length_x*length_y,(2*self.sample_size_x+1)*(2*self.sample_size_y+1))).astype(self.data_type)
        train_y = np.zeros((length_x*length_y,1))
        index = 0
        
        for i in range(length_x):
            for j in range(length_y):
                data = self.original_data[j:j+2*self.sample_size_x+1,i:i+2*self.sample_size_y+1].flatten()   
                train_x[index,:] = data              
                train_y[index]=self.labled_data[j+self.sample_size_x+1,i+self.sample_size_y+1] 
                index = index + 1
            
        print('data_have_been_generated')
        
        train_y = (train_y /255).astype(self.data_type)
        
        return train_x, train_y


class DataLoader_for_predict():
    def __init__(self, original_picture, sample_size_x = 5,sample_size_y = 5, data_type = 'uint16'):

        self.original_data = mpimg.imread(original_picture)
        self.size_X = len(self.original_data[0,:])
        self.size_Y = len(self.original_data[:,0])
        self.sample_size_x = sample_size_x
        self.sample_size_y = sample_size_y
        self.data_type = data_type
    
    def generate_predict_data(self):      
        length_x = self.size_X - 2*self.sample_size_x -1
        length_y = self.size_Y - 2*self.sample_size_y -1    
        test_x = np.zeros((length_x*length_y,(2*self.sample_size_x+1)*(2*self.sample_size_y+1)))
        index = 0
        
        for i in range(length_x):
            for j in range(length_y):
                data = self.original_data[j:j+2*self.sample_size_x+1,i:i+2*self.sample_size_y+1].flatten()   
                test_x[index,:] = data              
                index = index + 1
        return test_x.astype(self.data_type)
    
    def generate_predict_lable(self,model):
        length_x = self.size_X - 2*self.sample_size_x -1
        length_y = self.size_Y - 2*self.sample_size_y -1
        test_x = self.generate_predict_data()
        test_y = model.predict(test_x)
        test_y = np.argmax(test_y,axis = 1).reshape((length_x,length_y))
        #test_y = test_y[:,0].reshape((length_x,length_y))
        return test_y.T
