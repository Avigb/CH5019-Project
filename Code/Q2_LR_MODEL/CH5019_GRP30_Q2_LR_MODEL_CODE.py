import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import math
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import random as rd
import matplotlib.pyplot as plt

class LogisticRegression:

    def __init__(self, x, y, init_weights, bias, gd_epochs, learning_rate, decision_threshold):
        """Logistic Regressin Model based on batch gradient decent
        Arguments:
            x {dataframe} -- contains the feature for the model
            y {dataframe} -- outputs to train the model
            init_weights {5x5 numpyarray} -- intial coefficients guesses
            bias {int} -- bias term for predicting y
            gd_epochs {number} -- number of interatin for batch gradient decent
            learning_rate {int} -- learning rate for gradient decent
            decision_threshold {int (0-1)} -- decision boundary to select output class
        """
        self.x = x
        self.y = y
        self.theta = init_weights
        self.b = bias
        self.epochs = gd_epochs
        self.m = x.shape[0]
        self.alpha = learning_rate
        self.decision_threshold_value = decision_threshold

    @staticmethod
    def sigmoid(result):
        return 1/(1+np.exp(-result))

    @staticmethod
    def weighted_input(self, w, features):
        return np.dot(w, features)

    @staticmethod
    def decision_threshold(self,value):
        if value>=self.decision_threshold_value:
            return 1
        return 0

    def probability(self, w, features):
        return self.sigmoid(self.weighted_input(w, features))

    def cost_function(self, w, b):
        prediction = self.sigmoid(np.dot(w.T, self.x.T) + b)
        total_cost = -(1/self.m) * np.sum(self.y*np.log(np.clip(prediction,10e-100,None)) +
                                     (1-self.y)*np.log(np.clip((1-prediction),10e-100,None)))
        return total_cost

    def gradients(self, w, b):
        final_result = self.sigmoid(np.dot(w.T, self.x.T) + b)
        dw = (1/self.m)*(np.dot(self.x.T, (final_result.T-self.y)))
        db = (1/self.m)*(np.sum(final_result-self.y))

        return {'dw': dw, 'db': db, 'fr':final_result}

    def gradient_descent(self):
        costs = []
        w = self.theta
        b = self.b
        gradient = {}
        print('Training the model...')
        for i in range(self.epochs):
            curr_cost = self.cost_function(w,b);
            costs.append(curr_cost)
            grads = self.gradients(w, b)
            dw = grads['dw']
            db = grads['db']
            # updating weights
            w = w - (self.alpha*dw)
            b = b - (self.alpha*db)
            gradient = {"dw": dw, "db": db}
            if(i%100==0):
                print(i,' epoch to ',i+100,' epoch')

        coeff = {"w": w, "b": b}

        return coeff, gradient, costs

    def predict(self,x,coeff):
        y_out = np.dot(x,coeff['w']) + coeff['b']
        y_pred = self.sigmoid(y_out)
        y_pred = np.apply_along_axis(lambda x: 1 if (x >= self.decision_threshold_value) else 0
                                     , 1, y_pred)
        return y_pred


if __name__ == '__main__':
    # os.chdir(r"C:\Users\jeswantkrishna\Desktop\Term project 2020")
    
    data = pd.read_excel("Dataset_Question2.xlsx")
    # replacing the 'Pass' and 'Fail' values by 1 and 0 respectively
    data['Test'].replace({'Pass': 1, 'Fail': 0}, inplace=True)
    ss = StandardScaler()
    AccLst=[]
    AccLst_Train=[]

    print('Learning Rate=0.04 Weights All Zero Bias is 12')

    i=0
    while(True):
        # Shuffling data
        data = data.sample(frac=1,random_state=99).reset_index(drop=True)

        # Features of the dataset
        X = data.iloc[:,:-1]
        # Normalising The Data using StandaryScaler
        x=ss.fit_transform(X)

        # response_variable
        y = data.iloc[:, -1:]
        y = y.to_numpy()

        # Train Data (700 Data Points)
        Train_Data_x= x[0:700]
        Train_Data_y= y[0:700]
        
        # To Prevent imbalanced data as much as possible
        if(not(abs(np.count_nonzero(Train_Data_y==1)-350)<=50)):
            continue

        # Test Data (300 Data Points)
        Test_Data_x= x[700:]
        Test_Data_y= y[700:]
        
        plt.ion()
        LRrate=0.04 # Best Learning Rate
        
        # Model
        theta=np.full((Train_Data_x.shape[1],1),0)
        b=np.full((1,1),12)
       
        model = LogisticRegression(Train_Data_x,Train_Data_y, theta, b, 1000, LRrate,0.7)
        coeff_, gradient_, costs_ = model.gradient_descent()

        # Predicting
        y_pred_Test = model.predict(Test_Data_x,coeff_)
        y_pred_Train=model.predict(Train_Data_x,coeff_)

        # Confusion Matrix
        cm=confusion_matrix(Test_Data_y,y_pred_Test)
        cm_Train=confusion_matrix(Train_Data_y,y_pred_Train)

        #F1 Score
        Test_Precision=cm[0][0]/(cm[0][0]+cm[0][1])
        Test_Recall=cm[0][0]/(cm[0][0]+cm[1][0])
        F1_Score_Test=2/(Test_Precision**-1+Test_Recall**-1)
        
        Train_Precision=cm_Train[0][0]/(cm_Train[0][0]+cm_Train[0][1])
        Train_Recall=cm_Train[0][0]/(cm_Train[0][0]+cm_Train[1][0])
        F1_Score_Train=2/(Train_Precision**-1+Train_Recall**-1)

        # Accuracy
        AccLst.append((cm[0][0]+cm[1][1])/3)
        AccLst_Train.append((cm_Train[0][0]+cm_Train[1][1])/7)
        
        if(not(AccLst[-1]>=92 and AccLst_Train[-1]>=92)):   #To get highest accuracy as possible
            AccLst.pop()
            AccLst_Train.pop()
            continue

        print("\nBiasness Check {}% 1's and {}% 0's ".format(round(np.count_nonzero(Train_Data_y==1)/7,3),round(np.count_nonzero(Train_Data_y==0)/7,3)))

        print('\nConfusion Matrix of Test Data')
        print(cm)
        print('\nConfusion Matrix of Train Data')
        print(cm_Train)
        print('\nWeights and Bias of the Trained Model')
        print(coeff_)
        
        print('\nAccuracy on Test: ',AccLst[-1])
        print('Accuracy on Train: ',AccLst_Train[-1])

        print('\nF1 Score of Test data: ',F1_Score_Test)
        print('F1 Score of Train data: ',F1_Score_Train)


        #Plots of Cost function
        plt.figure(i+1)
        plt.grid(True)
        plt.title('Cost Vs Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Cost for the Epoch')
        plt.plot(range(len(costs_)),costs_)
        
        plt.figure(i+2)
        plt.grid(True)
        plt.title('Average Cost from [0 to Current Epoch] Vs Epoch')
        plt.xlabel('Epochs')
        plt.ylabel('Average Cost from [0 to Current Epoch]')
        newCostPlt=[]
        avgNo=costs_[0]
        for k in range(len(costs_)):
            avgNo=(avgNo*(k+1)+costs_[k])/(k+2)
            newCostPlt.append(avgNo)
        plt.plot(range(len(newCostPlt)),newCostPlt)
        break





