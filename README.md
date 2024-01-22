# NeuraCast - RNN Model

<samp>Modeling patient time series data with Long Short Term Memory Recurrent Neural Network in keras. Provides probablity of patient survival using electronic health records. Data is preprocessed filling gaps, normalizing features, and padding. A single LSTM using binary cross entropy loss and RMSprop optimizer.</samp>

## Tools/Frameworks

- `Python` : Primary language for writing neural network model and components
- `Keras` : Neural network API used to define model architecture
- `Tensorflow` : Backend for keras handling NN executions
- `Pandas` : Dataframe used to interact with patient data and preprocessing

## Background
- Project utilizes electronic health records from Children's Hospital Los Angeles to predict patient mortality
- 5000 patients in training sets with multiple observations each of which including 265 categories of metrics
- Metrics include heart rate, creatinine, 02, dopamine, etc
- Not all values have been recorded for patients; missing data poitns

## Model Choices

<table>
  <tr>
    <td width="33%"">
    <samp>Binary Cross Entropy</samp>
    </td>
    <td width="66%">
    <samp>Calculates difference between calculated and actual results; Optimal cost function for classification problems</samp>
    </td>
  </tr>
  <tr>
    <td width="33%"">
    <samp>0.005 Learning Rate</samp>
    </td>
    <td width="66%">
    <samp>Balances training and epoch duration with step size for ideal model learning time</samp>
    </td>
  </tr>
  <tr>
    <td width="33%"">
    <samp>Dropout Parameter</samp>
    </td>
    <td width="66%">
    <samp>Random dropout of input vector elements; forces model to use covariate variables as proxy</samp>
    </td>
  </tr>
</table>

## Notes
- Model completed as part of NVDIA Deep Learning Institute Course
