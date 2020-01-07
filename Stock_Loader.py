import pandas as pd
import json
from datetime import datetime

def json_to_df(stock_data):

    idx = stock_data.find(".")
    suffix = stock_data[idx+1:]

    # eg) intraday_30.json -> "intraday"
    if stock_data.find("_") != -1:
        idx = stock_data.find("_")

    assert type(stock_data) == str, "Input must be string (Example: 'Stock_data.txt')"
    assert suffix == "json", "Input file must be json file"

    import json

    category_dict = {"Date": [], "open": [], "close": [], "high": [], "low": [], "volume": []}

    with open(stock_data, "r") as stock_js:
        data = json.load(stock_js)
        for day in data[stock_data[:idx]]:
            category_dict["Date"].append(day)
            for category in data[stock_data[:idx]][day]:
                category_dict[category].append(data[stock_data[:idx]][day][category])

    df_stock = pd.DataFrame(category_dict)
    df_stock = df_stock[::-1]
    df_stock = df_stock.reset_index(drop=True)

    return df_stock

def select_appl_stock(df_stock):

    df_date = df_stock.iloc[:,  0]
    df_appl = df_stock.iloc[:, 16:21]
    df_appl = pd.concat([df_date, df_appl], axis=1)
    df_appl = df_appl.drop(df_appl.index[0:2])
    df_appl = df_appl.reset_index(drop=True)
    df_appl = df_appl.rename(columns={"Unnamed: 0": "TimeStamp",
                                      'AAPL': "Open", 'AAPL.1': "High",'AAPL.2': "Low",
                                      'AAPL.3': "Close", 'AAPL.4': "Volume"})

    return df_appl

def date_converter(df, date_format):

    df["Year"] = None
    df["Month"] = None
    df["Day"] = None
    df["Hour"] = None
    df["Day_of_Week"] = None

    from datetime import datetime

    for idx, row in enumerate(df["TimeStamp"]):
        obj = datetime.strptime(row, date_format)
        df["Year"].loc[idx] =  obj.year
        df["Month"].loc[idx] =  obj.month
        df["Day"].loc[idx] =  obj.day
        df["Hour"].loc[idx] =  obj.hour
        df["Day_of_Week"].loc[idx] = obj.weekday()

    return df

def load_data(dataset_name):
    
    print("...Downloading Datasets")
    df_sp = pd.read_csv(dataset_name)
    df_appl = select_appl_stock(df_sp)
    
    print("...Adding Values to Datasets")
    df_added = date_converter(df_appl, "%Y-%m-%d %H:%M:%S")
    
    print("...Done !!")

    return df_added