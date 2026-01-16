dataset available from ![Here](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)

## Data analysis
Dataset contains 25979 rows with missing data points   
There is never a situation where only one feature is missing from a field  
If a feature is missing all the features (not including the date and time) are missing from a datapoint

82 days contain null data points

Dataset stores Dates and Time of days seperately, so for pandas they will need to be combined to work as a datetime object

## How data is changed
Columns, Date and Time have been combined to be able to work with the pandas datetime object