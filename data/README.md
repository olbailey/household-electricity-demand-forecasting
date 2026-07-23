# How to run
## Downloading Dataset
run `python(3) scripts/download_data.py` in the main project directory to download dataset into directory

## Pre-processing Dataset
run `python(3) scripts/preprocess_data.py` in the main project directory to process the dataset, 
and create the working .parquet dataset the model datasets work from


dataset Website source ![Here](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)



## Data Description
1. date: Date in format dd/mm/yyyy
2. time: time in format hh:mm:ss
3. global_active_power: household global minute-averaged active power (in Kilowatt)
4. global_reactive_power: household global minute-averaged reactive power (in Kilowatt)
5. voltage: minute-averaged voltage (in Volts)
6. global_intensity: household global minute-averaged current intensity (in Ampere)
7. sub_metering_1: energy sub-metering No. 1 (in watt-hour of active energy). It corresponds to the kitchen, containing mainly a dishwasher, an oven and a microwave (hot plates are not electric but gas powered).
8. sub_metering_2: energy sub-metering No. 2 (in watt-hour of active energy). It corresponds to the laundry room, containing a washing-machine, a tumble-drier, a refrigerator and a light.
9. sub_metering_3: energy sub-metering No. 3 (in watt-hour of active energy). It corresponds to an electric water-heater and an air-conditioner.

### Active power
The usable or consumed electrical energy in an AC circuit

### Reactive power
The alternating current flowing back and forth in an electrical circuit.

## Data analysis
Dataset contains 25979 rows with missing data points   
There is never a situation where only one feature is missing from a field.
If a feature is missing all the features (not including the date and time) are missing from a datapoint.

82 days contain null data points

Dataset stores Dates and Time of days seperately, so for Pandas they will need to be combined to work as a datetime object.

First data point is at 17:24 on the 16th of December 2006
Last data point is at 21:02 on the 26th of November 2010
Data is collected every minute

Data follows a familiar pattern each year, so should be possible to learn long term trends

## How data is changed
Columns, Date and Time have been combined to be able to work with the pandas datetime object