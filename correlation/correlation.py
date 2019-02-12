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

cols = []

for i, col1 in enumerate(factor_return):
    print(col1)
    cols.append(col1)
    for j, col2 in enumerate(factor_return):
        r, p = pearsonr(factor_return[col1], factor_return[col2])
        result[i, j] = r


print(result)

corr_df = pd.DataFrame(data=result, index=cols, columns=cols)

corr_df.to_csv("factor_return_correlation.csv")

