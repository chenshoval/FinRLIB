import time
import threading
from ib_insync import IB, Stock, util
# Create an IB instance
ib = IB()

# Connect to IB Gateway for paper trading
# Use '127.0.0.1' if the IB Gateway is running on the same machine or VM
# Use port '7497' for paper trading
ib.connect('127.0.0.1', 4002, clientId=0)

# Define the contract
contract = Stock('AAPL', 'SMART', 'USD')
ticker = ib.reqMktData(contract, "", False, False, [])
time.sleep(5)
print(util.df([ticker]))

history = ib.reqHistoricalData(contract, "", "10 D", "1 min", "TRADES", 1, 1, False, [])
time.sleep(5)
print(history)

account_details = ib.reqAccountUpdates("")
time.sleep(5)
print(account_details)

# Disconnect from IB Gateway
ib.disconnect()
print("done!")