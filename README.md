Notes:

Time series (sequential data) - data points orded in time. 
Multivariate - multiple variables

Complex time series data has multivariate points. e.g financial markets, health records,  etc

- goal is to predict a variable value

  will use deep neural network to predict patient mortality using timeseries data from patient records. provides
  a analystic fremwork for medical professionals to predict patient mortality ahead of time and in adjusting possible sollutions

  using EHR (electronic health record) data from childrens hopsital los angeles (CHLA)
  
RNN - recurrent neural network, well suited for sequential data
- limited to only looking back a frew steps due to vanishing gradient
LSTM long short-term memory networks
- solves vanishing gradient problem using memory cells
LSTM special type of RNN

steps: 
visualize time series data
prepare data for modeling
create NN structure using keras
compare model quality

LSTM has strong memory


inputs: patient physiology
ouputs patient mortaility chance

keras neural network api in python

pandas using data frame object for data manipulation
- reading and writing data between different Data structures

numpy scienfitic package suppoting arrays and matricies

matplotlib 2d plotting library

5000 using patients in training set
each patient has multiple observations each of which as 265 categories

some measuremnets include
heart rate, gender, creatinine, 02, dopamine

not all mesaumrents taken for all patients; some missing values

rows: observations
columns: variables

normalize data
fill gaps
pad and trundace data sequence
architect LSTM RNN with KERAS. train model. evaluate using vailation testset. visualize. compare to baseline

comparisons PRISM3 PIM2 with model


0 - dead
1 - alive




keras can use CPU and GPUs
GPUs more power efficient and fast made for parallel processing