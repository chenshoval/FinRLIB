from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import threading
import time
import numpy as np
import datetime
from ibapi.order import Order

class IBKRApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []  # Initialize a list to store OHLCV data

    def openOrder(self, orderId, contract, order, orderState):
        print(f"Order ID: {orderId}, Symbol: {contract.symbol}, Order Type: {order.orderType}, Quantity: {order.totalQuantity}")
        self.open_order_ids.append(orderId)

    def nextValidId(self, orderId):
        self.nextOrderId = orderId
        self.reqOpenOrders()

    def cancel_open_orders(self):
        for order_id in self.open_order_ids:
            print(f"Cancelling Order ID: {order_id}")
            self.cancelOrder(order_id)

    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId}, {errorCode}, {errorString}")

    def tickPrice(self, reqId, tickType, price, attrib):
        print(f"Tick Price. Ticker Id: {reqId}, tickType: {tickType}, Price: {price}")

    def updateAccountValue(self, key, val, currency, accountName):
        balance = float(val) if key == "FullAvailableFunds" else None
        print(f"Account Update. Key: {key}, Value: {val}, Currency: {currency}, Account: {accountName}")

    def historicalData(self, reqId, bar):
        print(f"HistoricalData. ReqId: {reqId}, Date: {bar.date}, Open: {bar.open}, High: {bar.high}, Low: {bar.low}, Close: {bar.close}, Volume: {bar.volume}")
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])

    def currentTime(self, time: int):
        print(f"Current Time: {time}")
        self.current_time = datetime.datetime.fromtimestamp(time).astimezone(datetime.timezone.utc)
        print(f"Current Time datetime: {self.current_time}")
        
    def historicalDataEnd(self, reqId, start, end):
        print(f"HistoricalDataEnd. ReqId: {reqId}, from {start} to {end}")
        self.data = np.array(self.data)
        self.disconnect()

    def start(self):
        self.reqCurrentTime()

    def is_market_open(self):
        # Define market open and close times (example: 9:30 AM to 4:00 PM EST)
        market_open = datetime.time(14, 30, tzinfo=datetime.timezone.utc)
        market_close = datetime.time(21, 0, tzinfo=datetime.timezone.utc)
        current_time = self.current_time.timetz()
        current_day = self.current_time.weekday()  # Monday is 0 and Sunday is 6
        return  current_day < 5 and market_open <= current_time <= market_close

def awaitMarketOpen(ibApp):
    ibApp.reqCurrentTime()
    time.sleep(3)
    while not ibApp.is_market_open():
        ibApp.reqCurrentTime()
        time.sleep(1)  # Wait for the current time to be updated
        if ibApp.current_time:
            current_time = ibApp.current_time
            current_day = current_time.weekday()
            opening_time = datetime.time(14, 30, tzinfo=datetime.timezone.utc)
            
            # Calculate the next opening time
            if current_day >= 5:  # If it's Saturday (5) or Sunday (6)
                days_until_monday = 7 - current_day
                next_opening_date = current_time.date() + datetime.timedelta(days=days_until_monday)
            elif current_time.timetz() >= opening_time:
                # If it's after the market opening time today, set to next weekday
                next_opening_date = current_time.date() + datetime.timedelta(days=1)
                if next_opening_date.weekday() >= 5:  # If it's Saturday or Sunday
                    days_until_monday = 7 - next_opening_date.weekday()
                    next_opening_date += datetime.timedelta(days=days_until_monday)
            else:
                # If it's before the market opening time today
                next_opening_date = current_time.date()

            next_opening_datetime = datetime.datetime.combine(next_opening_date, opening_time)
            time_to_open = int((next_opening_datetime - current_time).total_seconds() / 60)
            print(f"{time_to_open} minutes until market open.")
        time.sleep(60)
    print("Market is now open.")

def run_loop():
    app.run()

app = IBKRApp()
app.connect("127.0.0.1", 7497, 0)

api_thread = threading.Thread(target=run_loop, daemon=True)
api_thread.start()

time.sleep(3)  # Sleep interval to allow time for connection to server

contract = Contract()
contract.symbol = "AAPL"
contract.secType = "STK"
contract.exchange = "SMART"
contract.currency = "USD"

# Request account updates
app.reqAccountUpdates(True, "")

app.reqMarketDataType(1)
app.reqMktData(1, contract, "", False, False, [])
#app.reqHistoricalData(1, contract, "", "10 D", "1 min", "TRADES", 1, 1, False, [])

time.sleep(10)  # Sleep to allow enough time to receive the data

#awaitMarketOpen(app)
# Create an order
order = Order()
order.action = "BUY"
order.orderType = "MKT"
order.totalQuantity = 10

# Place the order
app.placeOrder(app.nextOrderId, contract, order)

print(len(app.data))
print(app.data)
app.disconnect()
