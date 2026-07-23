# Model Configs
This is a guide on how to use .yaml config files to architectural structures and hyperparameter tuning for the pytorch I have created.

## Selecting the files
In the `scr/training/training.py` file there is a section for selecting the yaml file configs.
```python
with open("configs/config.yaml", 'r') as file:
    configs = yaml.safe_load(file)
```
Replace the file name you create or are working with instead of the config.yaml file within the string. Make sure to keep config files within this directory.

# Config Features
## Model
This does slightly differ between LSTM and MLP Models.
### Type
The name for the type of model, used in training.py to select which model object to create, must match expected options, or option must be created.
```python
if configs["model"]["type"] == "MultiLayerPerceptron":
    print(f"Training MultiLayerPerceptron model...")
    model = MLP(configs["model"], WINDOW_SIZE)

elif configs["model"]["type"] == "LSTM":
    print(f"Training LSTM model...")
    model = LSTM(configs["model"])
```
### Input Size
defines how many features each data point has, must be consistent throughout dataset. For MLP this is not the number of input neurons as this is used in conjuction with window size to calculate input nuerons for the model.

### Hidden Layers
For MLP this is a list of the number of neurons per hidden layer to be constructed. For LSTM it is an integer number for the number for hidden layers for it to contain. 

### Output Size
Number of datapoints the model has to forecast past the datapoints from the window it is provided

### LSTM Hidden Size
The size of the hidden layers with the LSTM architecture

### MLP Activation
The activation function used to provide non-linearity to the model, can choose between 'relu' and 'tanh'

## Training
### Batch Size
Batch size the DataLoader feeds the model. Larger values tend to increase efficiency though must also match GPU capabilities

### Learning Rate
The Float fed to the Optimizer to control how large a step the model takes when updating its weights during traing 

### Window Size
The number of concurrent datapoints provided to the model per sample

### Stride Size
The gap between each samples starting datapoint. Necessary for performance and efficiency with large window sizes

### Patience - Early Stopping
The number of epochs where the training has stagnated until the training process is terminated and the best parameters are restored

### Min Delta - LR Schedular & Early Stopping
The floating point value for to what degree the tracked metric has to decrease to count as an improvement. (By default it tracks MAE, Mean Absolute Error of the validation dataset)

## Data
### Model Directory
The path for the saved model parameters to be stored. Suggested path within the main directory would be
`Outputs/models/enter_name`