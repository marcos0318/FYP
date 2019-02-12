import rqdatac as rq
import pandas as pd

# initializing rqdatac
rq.init()

start_date = "2016-01-01"
end_date = "2018-12-30"

factor_return = rq.get_factor_return(start_date, end_date, factor="all", universe="whole_market", method="explicit")

factor_return.to_csv("factor return")