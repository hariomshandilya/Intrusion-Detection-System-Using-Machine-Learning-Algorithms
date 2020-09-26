# Intrusion-Detection-System-Using-Machine-Learning-Algorithms
Problem Statement: The task is to build a network intrusion detector, a predictive model capable of distinguishing between bad connections, called intrusions or attacks, and good normal connections.  Introduction: Intrusion Detection System is a software application to detect network intrusion using various machine learning algorithms.IDS monitors a network or system for malicious activity and protects a computer network from unauthorized access from users, including perhaps insider. The intrusion detector learning task is to build a predictive model (i.e. a classifier) capable of distinguishing between ‘bad connections’ (intrusion/attacks) and a ‘good (normal) connections’.  Attacks fall into four main categories: #DOS: denial-of-service, e.g. syn flood; #R2L: unauthorized access from a remote machine, e.g. guessing password; #U2R: unauthorized access to local superuser (root) privileges, e.g., various “buffer overflow” attacks; #probing: surveillance and another probing, e.g., port scanning.

kddcup.names : A list of features.
kddcup.data.gz : The full data set
kddcup.data_10_percent.gz : A 10% subset.
kddcup.newtestdata_10_percent_unlabeled.gz
kddcup.testdata.unlabeled.gz
kddcup.testdata.unlabeled_10_percent.gz
corrected.gz : Test data with corrected labels.
training_attack_types : A list of intrusion types.
typo-correction.txt : A brief note on a typo in the data set that has been corrected
feature name	description	type
duration	length (number of seconds) of the connection	continuous
protocol_type	type of the protocol, e.g. tcp, udp, etc.	discrete
service	network service on the destination, e.g., http, telnet, etc.	discrete
src_bytes	number of data bytes from source to destination	continuous
dst_bytes	number of data bytes from destination to source	continuous
flag	normal or error status of the connection	discrete
land	1 if connection is from/to the same host/port; 0 otherwise	discrete
wrong_fragment	number of “wrong” fragments	continuous
urgent	number of urgent packets	continuous
Table 1: Basic features of individual TCP connections.

feature name	description	type
hot	number of “hot” indicators	continuous
num_failed_logins	number of failed login attempts	continuous
logged_in	1 if successfully logged in; 0 otherwise	discrete
num_compromised	number of “compromised” conditions	continuous
root_shell	1 if root shell is obtained; 0 otherwise	discrete
su_attempted	1 if “su root” command attempted; 0 otherwise	discrete
num_root	number of “root” accesses	continuous
num_file_creations	number of file creation operations	continuous
num_shells	number of shell prompts	continuous
num_access_files	number of operations on access control files	continuous
num_outbound_cmds	number of outbound commands in an ftp session	continuous
is_hot_login	1 if the login belongs to the “hot” list; 0 otherwise	discrete
is_guest_login	1 if the login is a “guest”login; 0 otherwise	discrete
Table 2: Content features within a connection suggested by domain knowledge.

feature name	description	type
count	number of connections to the same host as the current connection in the past two seconds	continuous
Note: The following features refer to these same-host connections.	
serror_rate	% of connections that have “SYN” errors	continuous
rerror_rate	% of connections that have “REJ” errors	continuous
same_srv_rate	% of connections to the same service	continuous
diff_srv_rate	% of connections to different services	continuous
srv_count	number of connections to the same service as the current connection in the past two seconds	continuous
Note: The following features refer to these same-service connections.	
srv_serror_rate	% of connections that have “SYN” errors	continuous
srv_rerror_rate	% of connections that have “REJ” errors	continuous
srv_diff_host_rate	% of connections to different hosts	continuous
Various Algorithms Applied: Guassian Naive Bayes, Decision Tree, Random Fprest, Support Vector Machine, Logistic Regression.

Approach Used: I have applied various classification algorithms that are mentioned above on the KDD dataset and compare there results to build a predictive model.

Step 1 – Data Preprocessing:

Code: Importing libraries and reading features list from ‘kddcup.names’ file.

filter_none
import os 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import time 
  
# reading features list 
with open("..\\kddcup.names", 'r') as f: 
    print(f.read()) 
    
cols ="""duration, 
protocol_type, 
service, 
flag, 
src_bytes, 
dst_bytes, 
land, 
wrong_fragment, 
urgent, 
hot, 
num_failed_logins, 
logged_in, 
num_compromised, 
root_shell, 
su_attempted, 
num_root, 
num_file_creations, 
num_shells, 
num_access_files, 
num_outbound_cmds, 
is_host_login, 
is_guest_login, 
count, 
srv_count, 
serror_rate, 
srv_serror_rate, 
rerror_rate, 
srv_rerror_rate, 
same_srv_rate, 
diff_srv_rate, 
srv_diff_host_rate, 
dst_host_count, 
dst_host_srv_count, 
dst_host_same_srv_rate, 
dst_host_diff_srv_rate, 
dst_host_same_src_port_rate, 
dst_host_srv_diff_host_rate, 
dst_host_serror_rate, 
dst_host_srv_serror_rate, 
dst_host_rerror_rate, 
dst_host_srv_rerror_rate"""
columns =[] 
for c in cols.split(', '): 
    if(c.strip()): 
       columns.append(c.strip()) 
  
columns.append('target') 
print(len(columns)) 
with open("..\\training_attack_types", 'r') as f: 
    print(f.read()) 
Code: Creating a dictionary of attack_types

filter_none
brightness_4
   
attacks_types = { 
    'normal': 'normal', 
'back': 'dos', 
'buffer_overflow': 'u2r', 
'ftp_write': 'r2l', 
'guess_passwd': 'r2l', 
'imap': 'r2l', 
'ipsweep': 'probe', 
'land': 'dos', 
'loadmodule': 'u2r', 
'multihop': 'r2l', 
'neptune': 'dos', 
'nmap': 'probe', 
'perl': 'u2r', 
'phf': 'r2l', 
'pod': 'dos', 
'portsweep': 'probe', 
'rootkit': 'u2r', 
'satan': 'probe', 
'smurf': 'dos', 
'spy': 'r2l', 
'teardrop': 'dos', 
'warezclient': 'r2l', 
'warezmaster': 'r2l', 
} 
Code: Reading the dataset(‘kddcup.data_10_percent.gz’) and adding Attack Type feature in the training dataset where attack type feature has 5 distinct values i.e. dos, normal, probe, r2l, u2r.

filter_none
brightness_4
path = "..\\kddcup.data_10_percent.gz"
df = pd.read_csv(path, names = columns) 
  
# Adding Attack Type column 
df['Attack Type'] = df.target.apply(lambda r:attacks_types[r[:-1]]) 
df.head() 
 Shape of dataframe and getting data type of each feature

filter_none
brightness_4
  
df.shape 
Finding missing values of all features.

filter_none
brightness_4
df.isnull().sum()
Code: Finding Categorical Features

filter_none
brightness_4
# Finding categorical features 
num_cols = df._get_numeric_data().columns 
  
cate_cols = list(set(df.columns)-set(num_cols)) 
cate_cols.remove('target') 
cate_cols.remove('Attack Type') 
  
cate_cols 
Code: Data Correlation – Find the highly correlated variables using heatmap and ignore them for analysis.

filter_none
brightness_4
  
  
df = df.dropna('columns')# drop columns with NaN 
  
df = df[[col for col in df if df[col].nunique() > 1]]# keep columns where there are more than 1 unique values 
  
corr = df.corr() 
  
plt.figure(figsize =(15, 12)) 
  
sns.heatmap(corr) 
  
plt.show() 
Code:

filter_none
brightness_4
  
  
# This variable is highly correlated with num_compromised and should be ignored for analysis. 
#(Correlation = 0.9938277978738366) 
df.drop('num_root', axis = 1, inplace = True) 
  
# This variable is highly correlated with serror_rate and should be ignored for analysis. 
#(Correlation = 0.9983615072725952) 
df.drop('srv_serror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9947309539817937) 
df.drop('srv_rerror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with srv_serror_rate and should be ignored for analysis. 
#(Correlation = 0.9993041091850098) 
df.drop('dst_host_srv_serror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9869947924956001) 
df.drop('dst_host_serror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with srv_rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9821663427308375) 
df.drop('dst_host_rerror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9851995540751249) 
df.drop('dst_host_srv_rerror_rate', axis = 1, inplace = True) 
  
# This variable is highly correlated with srv_rerror_rate and should be ignored for analysis. 
#(Correlation = 0.9865705438845669) 
df.drop('dst_host_same_srv_rate', axis = 1, inplace = True) 
Code: Feature Mapping – Apply feature mapping on features such as : ‘protocol_type’ & ‘flag’.

filter_none
brightness_4
  
# protocol_type feature mapping 
pmap = {'icmp':0, 'tcp':1, 'udp':2} 
df['protocol_type'] = df['protocol_type'].map(pmap) 

  
# flag feature mapping 
fmap = {'SF':0, 'S0':1, 'REJ':2, 'RSTR':3, 'RSTO':4, 'SH':5, 'S1':6, 'S2':7, 'RSTOS0':8, 'S3':9, 'OTH':10} 
df['flag'] = df['flag'].map(fmap)
Code: Remove irrelevant features such as ‘service’ before modelling

filter_none
brightness_4
  
df.drop('service', axis = 1, inplace = True) 
Step 2 – Modelling

Code: Importing libraries and splitting the dataset

filter_none
brightness_4
  
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler 
Code:

filter_none
brightness_4
  
# Splitting the dataset 
df = df.drop(['target', ], axis = 1) 
print(df.shape) 
  
# Target variable and train set 
y = df[['Attack Type']] 
X = df.drop(['Attack Type', ], axis = 1) 
  
sc = MinMaxScaler() 
X = sc.fit_transform(X) 
  
# Split test and train data  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42) 
print(X_train.shape, X_test.shape) 
print(y_train.shape, y_test.shape) 
Code: Python implementation of Guassian Naive Bayes

filter_none
brightness_4
  
# Gaussian Naive Bayes 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score 
  
clfg = GaussianNB() 
start_time = time.time() 
clfg.fit(X_train, y_train.values.ravel()) 
end_time = time.time() 
print("Training time: ", end_time-start_time)
