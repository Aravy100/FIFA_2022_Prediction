# importing packages

import pandas as pd
import numpy as np
import csv
import requests
import seaborn as sns
from matplotlib import pyplot as plt

# Loading the results of all past matches
results = pd.read_csv(r"C:\Users\17207\Desktop\fifa data\results.csv")
results['date']=pd.to_datetime(results['date'])
results = results[results['date']>"2005-12-31"] # filtering for results from 2006

# Reading in the fixtures for the 2022 FIFA World Cup
fixtures = pd.read_csv("https://fixturedownload.com/download/fifa-world-cup-2022-UTC.csv")
# Since we know that the group stage has only 48 matches, lets filter those out
teams = fixtures_raw[fixtures_raw['Match Number']<49]
# Let us filter the data for only the teams participating in the tournament 
x = np.array(teams.iloc[:,4].tolist() + teams.iloc[:,5].tolist())
teams = np.unique(x)

# Filtering results so that only teams in the fixtures data appear
pd.merge(pd.DataFrame(teams, columns=["teams"]),results["home_team"],
         how="inner",left_on="teams",right_on="home_team").drop_duplicates().reset_index(drop=True)
results_22 = results[(results['home_team'].isin(teams) | results['away_team'].isin(teams))]
         
# Cleaning and keeping names of Nations aligned
results = results.replace("United States","USA")
results = results.replace("South Korea","Korea Republic")

# Reading in rankings dataset:
ranking = pd.read_csv(r"C:\Users\17207\Desktop\fifa_ranking-2022-08-25.csv")

# It looks like this dataset is corrupt; that is the ranking is the same irrespective of the dates. 
# Upon further checks we see that most of the data is this way.
# Filtering for only the latest dates..
temp = max(ranking['rank_date'])
ranking = ranking[ranking['rank_date']==temp]

# Let us check if all the 32 countries we are interested are correctly represented in ranking
pd.merge(pd.DataFrame(teams, columns=["teams"]),ranking["country_full"],
         how="left",left_on="teams",right_on="country_full").drop_duplicates().reset_index(drop=True)
         
# Joining rankings with our previous dataset
df_t = pd.merge(results_22,ranking,how="left",left_on="home_team", right_on="country_full")
df = pd.merge(df_t,ranking,how="left",left_on="away_team", right_on="country_full")

# We now have the master dataframe of Rankings + Results + Fixtures
# Let us cleanup the rest of the country names 
ds = df[df['rank_x'].isnull()].drop_duplicates().reset_index()
missing = []
missing.extend(ds["home_team"].unique().tolist())
ds = df[df['rank_y'].isnull()].drop_duplicates().reset_index()
missing.extend(ds["away_team"].unique().tolist())
missing = np.unique(missing)
df.dropna(inplace=True)
df.shape
# (5267, 25)


# Outliers for Goals
f, axes = plt.subplots(1, 2)
sns.boxplot(y = df['home_score'],ax=axes[0])
sns.boxplot(y = df['away_score'],ax=axes[1])


# Filtering for Belgium National Team. This is for our Neural Network
# We will start simple by filtering our database for ONLY ONE COUNTRY. This will allows us to better understand the network and interpret the results
df_belgium = df[(df['home_team']=='Belgium')|(df['away_team']=='Belgium')].reset_index(drop=True)

# Splitting the dataframe into 2: 1 where Belgium played at Home and another where it is away
df_belgium_home = df_belgium[df_belgium["home_team"]=="Belgium"]
df_belgium_away = df_belgium[df_belgium["away_team"]=="Belgium"]
df_belgium_home = df_belgium_home.rename(columns={'away_team':'against_team',
                                                  'home_score':'score_belgium',
                                                 'away_score':'score_against',
                                                 'rank_x':'rank_belgium',
                                                 'rank_y':'rank_against',
                                                 'total_points_x':'points_belgium',
                                                 'total_points_y':'points_against'})
df_belgium_home=df_belgium_home.drop(columns=['home_team','rank_belgium','points_belgium']).reset_index(drop=True)
df_belgium_away = df_belgium_away.rename(columns={'home_team':'against_team',
                                                  'home_score':'score_against',
                                                 'away_score':'score_belgium',
                                                 'rank_x':'rank_against',
                                                 'rank_y':'rank_belgium',
                                                 'total_points_x':'points_against',
                                                 'total_points_y':'points_belgium'})
df_belgium_away=df_belgium_away.drop(columns=['away_team','rank_belgium','points_belgium']).reset_index(drop=True)
df_belgium_t = df_belgium_home.append(df_belgium_away)
df_belgium_t['result']= np.where(df_belgium_t['score_belgium']>df_belgium_t['score_against'],"Win",
                                 (np.where(df_belgium_t['score_belgium']==df_belgium_t['score_against'],"Draw","Lost")))

# Now we have a dataframe for Belgium matches only 

# Let us construct a pie chart and table to see how many matches Belgium has won
piedata = df_belgium_t.groupby(['result']).count().reset_index()
piedata = piedata[['result','date']]
piedata = piedata.rename(columns={'date':'count'})
piedata


#    result	count
#0	Draw	30
#1	Lost	39
#2	Win	    108



# Our label 'result' needs to be one hot encoded
df_belgium_t['result']= np.where(df_belgium_t['score_belgium']>df_belgium_t['score_against'],1,0)

# We will also need to calculate a numeric equivalent for the location where the match was played
df_belgium_t['home_flag']=np.where(df_belgium_t['country']=="Belgium",1,0)


# We have 176 matches, and so we will take the first half for training and predict the rest.
# Our NN will try to predict based on rank and points of the team Belgium plays against and hence we will focus on these
df_belgium_t = df_belgium_t[['rank_against','home_flag','result']]
df_belgium_train = df_belgium_t.iloc[:88]
df_belgium_test= df_belgium_t.iloc[89:]


####################################    NEURAL NETWORK      ####################################
X = np.array(df_belgium_train[['rank_against','home_flag']])

print("X is:\n", X)   

y = np.array(df_belgium_train[['result']])
#print(y.shape)
print("y is:\n", y) 
  

class NeuralNetwork(object):
    def __init__(self):
        
        # Structure of our NN
        self.InputNumColumns = 2  ## columns
        self.OutputSize = 1
        self.HiddenUnits = 3  ## one layer with h units
        self.n = 150  ## number of training examples, n
        
        # Weights & biases
        self.W1 = np.random.randn(self.InputNumColumns, self.HiddenUnits) # c by h  
        print(self.W1) 
        self.W2 = np.random.randn(self.HiddenUnits, self.OutputSize) # h by o 
        self.b = np.random.randn(self.OutputSize, self.HiddenUnits)        
        self.c = np.random.randn(1, self.OutputSize)
        
        self.GA=False ## set this to True if you want the 
        ## average gradient over all examples rather than
        ## the sum
        
    def FeedForward(self, X):
        print("FeedForward\n\n")
        self.z = (np.dot(X, self.W1)) + self.b 
        #X is n by c   W1  is c by h -->  n by h
        print("Z1 is:\n", self.z)
        
        self.h = self.Sigmoid(self.z) #activation function    shape: n by h
        print("H is:\n", self.h)
        self.z2 = (np.dot(self.h, self.W2)) + self.c# n by h  @  h by o  -->  n by o  
        print("Z2 is:\n", self.z2)
        output = self.Sigmoid(self.z2)  
        print("Y^ -  the output is:\n", output)
        return output
        
    def Sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))
    
    def BackProp(self, X, y, output):
        print("\n\nBackProp\n")
        self.LR = 0.1
        
        # Y^ - Y
        self.output_error = output - y    
        print("Y^ - Y\n", self.output_error)
        print("SIG Y^\n", self.Sigmoid(output, deriv=True))
        
        ##(Y^ - Y)(Y^)(1-Y^)
        self.output_delta = self.output_error * self.Sigmoid(output, deriv=True) 
        print("D_Error (Y^)(1-Y^)(Y^-Y) is:\n", self.output_delta)
        
        ##(Y^ - Y)(Y^)(1-Y^)(W2)
        self.D_Error_W2 = self.output_delta.dot(self.W2.T) #  D_Error times W2
        #print("W2 is\n", self.W2)
        #print(" D_Error times W2\n", self.D_Error_W2)
        
        ## (H)(1 - H) (Y^ - Y)(Y^)(1-Y^)(W2)
        self.H_D_Error_W2 = self.D_Error_W2 * self.Sigmoid(self.h, deriv=True) 
        ## Note that * will multiply respective values together in each matrix
                

        # Updating Weights and Biases
        ##  XT  (H)(1 - H) (Y^ - Y)(Y^)(1-Y^)(W2)
        self.X_H_D_Error_W2 = X.T.dot(self.H_D_Error_W2) ## this is dW1
        
        ## (H)T (Y^ - Y)(Y^)(1-Y^)
        self.h_output_delta = self.h.T.dot(self.output_delta) ## this is dW2
        
        if(self.GA=="True"):
            print("Using average gradient........\n")
            self.W1 = self.W1 - self.LR*(self.X_H_D_Error_W2/self.n)
            self.W2 = self.W2 - self.LR*(self.h_output_delta/self.n) ## average the gradients

        else: 
            print("Using sum gradient........\n")
            self.W1 = self.W1 - self.LR*(self.X_H_D_Error_W2) # c by h  adjusting first set (input -> hidden) weights
            self.W2 = self.W2 - self.LR*(self.h_output_delta) # adjusting second set (hidden -> output) weights

        
        print("The b biases before the update are:\n", self.b)
        self.b = self.b  - self.LR*self.H_D_Error_W2
        print("Updated bs are:\n", self.b)
        self.c = self.c - self.LR*self.output_delta
        print("The W1 is: \n", self.W1)
        print("The W1 gradient is: \n", self.X_H_D_Error_W2)
        print("The W2 gradient  is: \n", self.h_output_delta)
        print("The biases b gradient is:\n",self.H_D_Error_W2 )
        print("The bias c gradient is: \n", self.output_delta)
        
    def TrainNetwork(self, X, y):
        output = self.FeedForward(X)
        self.BackProp(X, y, output)
        return output
        
MyNN = NeuralNetwork()

TotalLoss=[]
AvgLoss=[]
Epochs=100
for i in range(Epochs): 
    print("\nRUN:\n ", i)
    output=MyNN.TrainNetwork(X, y)
   
    #print("The y is ...\n", y)
    print("The output is: ", output)
    print("Total Loss:", .5*(np.sum(np.square(output-y))))
    TotalLoss.append( .5*(np.sum(np.square(output-y))))
    
    print("Average Loss:", .5*(np.mean(np.square((output-y)))))
    AvgLoss.append(.5*(np.mean(np.square((output-y)))))
    
import matplotlib.pyplot as plt

fig1 = plt.figure()
ax = plt.axes()
x = np.linspace(0, 100, Epochs)
ax.plot(x, TotalLoss)    

fig2 = plt.figure()
ax = plt.axes()
x = np.linspace(0, 100, Epochs)
ax.plot(x, AvgLoss)  


# Testing our Neural Network with test data
X = np.array(df_belgium_test[['rank_against','home_flag']])
Y_HAT=MyNN.FeedForward(X)


# Let us calculate the confusion matrix
temp = np.round(output)-np.array(df_belgium_test[['result']])
CorrectGuess = len(temp[temp==0])
CorrectGuess
# 49

temp = np.round(output)
temp2 = np.array(df_belgium_test[['result']])
CorrectGuessWin = np.sum(temp[temp==temp2])
CorrectGuessWin
# 34

CorrectGuessNotWin = CorrectGuess-CorrectGuessWin
CorrectGuessNotWin
# 15


print("Accuracy of our model is: "+str(round(CorrectGuess/88*100,2)))
print("This is how many times our model would predict a win/no win if we asked it to guess for 100 matches")
# Accuracy of our model is: 55.68
# This is how many times our model would predict a win/no win if we asked it to guess for 100 matches

temp = np.round(output)-np.array(df_belgium_test[['result']])
PredWinButFalse = len(temp[temp==1])
PredWinButFalse
# 26

PredNotWinButFalse = len(temp[temp==-1])
PredNotWinButFalse
# 13