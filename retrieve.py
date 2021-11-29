"""Usage:
    python3 retrieve.py apiKey stockSymbol > /path/to/result/file.csv
"""
import requests

import os, sys
import datetime, time, csv

from typing import List

alphaVantageURL = "https://www.alphavantage.co"
alphaVantageQueryURL = alphaVantageURL + "/query"

apiKey = sys.argv[1]
stockSymbol = sys.argv[2]

def last2YearsCandles(symbol: str)->str:
    queryParams = {
        "apikey": apiKey,
        "function": "TIME_SERIES_INTRADAY_EXTENDED",
        "interval": "1min",
        "adjusted": False,
        "symbol": symbol
    }
    colNames = ["time", "open", "high", "low", "close", "volume"]
    result = ""
    callAmount = 0
    for year in [1, 2]:
        for month in range(1, 13):
            callAmount += 1
            queryParams["slice"] = "year{}month{}".format(year, month)
            resp = requests.get(alphaVantageQueryURL, params=queryParams)
            
            split = resp.text.splitlines()
            data = split[1: ]
            
            if year == 1 and month == 1:
                result += split[0] + os.linesep
            result += os.linesep.join(data) + os.linesep
            
            # Restriction is 5 calls per minute and 500 calls per day
            if callAmount >= 5:
                time.sleep(65)
                callAmount = 0
    return result

candles = last2YearsCandles(stockSymbol)
print(candles)