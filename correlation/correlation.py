import rqdatac as rq
import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

# initializing rqdatac
rq.init()

start_date = "2016-01-01"
end_date = "2018-12-30"

factor_return = rq.get_factor_return(start_date, end_date, "all", universe="whole_market", method="explicit")

factor_return.to_csv("factor_return.csv")
factor_return.to_json(("factor_return.json"))

# calculate the corrlation here

result = np.zeros((10, 10))

col1_index = 0
for col1 in factor_return:
    for col2 in factor_return:


