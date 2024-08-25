from __future__ import annotations
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

import numpy as np
import datetime
import threading
import time

from ibapi.contract import Contract
import threading
import time
import numpy as np

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import uuid # for unique ID
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from ibapi.order import Order

class IBKRClient(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.positions = []
        self.data = []  # Initialize a list to store OHLCV data
        self.open_order_ids = []
        self.open_orders = []
        self.open_orders_received = False

    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId}, {errorCode}, {errorString}")

    def tickPrice(self, reqId, tickType, price, attrib):
        print(f"Tick Price. Ticker Id: {reqId}, tickType: {tickType}, Price: {price}")

    def updateAccountValue(self, key, val, currency, accountName):
        balance = float(val) if key == "FullAvailableFunds" else None
        print(f"Account Update. Key: {key}, Value: {val}, Currency: {currency}, Account: {accountName}")
        print(f"Balance: {balance}")

    def historicalData(self, reqId, bar):
        print(f"HistoricalData. ReqId: {reqId}, Date: {bar.date}, Open: {bar.open}, High: {bar.high}, Low: {bar.low}, Close: {bar.close}, Volume: {bar.volume}")
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])

    def historicalDataEnd(self, reqId, start, end):
        print(f"HistoricalDataEnd. ReqId: {reqId}, from {start} to {end}")
        self.data = np.array(self.data)
        self.disconnect()

    def openOrder(self, orderId, contract, order, orderState):
        print(f"Order ID: {orderId}, Symbol: {contract.symbol}, Order Type: {order.orderType}, Quantity: {order.totalQuantity}")
        self.open_order_ids.append(orderId)
        self.open_orders.append(order)

    def nextValidId(self, orderId):
        self.nextOrderId = orderId
        self.reqOpenOrders()

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        self.positions.append((contract, position, avgCost))

    def openOrderEnd(self):
        print("All open orders received.")
        self.open_orders_received = True
 
    def cancel_open_orders(self):
        for order_id in self.open_order_ids:
            print(f"Cancelling Order ID: {order_id}")
            self.cancelOrder(order_id)
        self.open_orders_received = False
        self.open_order_ids = []

    def start(self):
        self.reqCurrentTime()

    def is_market_open(self):
        # Define market open and close times (example: 9:30 AM to 4:00 PM EST those are the times in utc)
        market_open = datetime.time(14, 30, tzinfo=datetime.timezone.utc)
        market_close = datetime.time(21, 0, tzinfo=datetime.timezone.utc)
        current_time = self.current_time.timetz()
        current_day = self.current_time.weekday()  # Monday is 0 and Sunday is 6
        return  current_day < 5 and market_open <= current_time <= market_close
    
    def calulateCloseTime(self):
        # Define market close time (example: 4:00 PM EST)
        market_close = datetime.time(21, 0, tzinfo=datetime.timezone.utc)
        current_time = self.current_time.timetz()
        time_to_close = (datetime.datetime.combine(datetime.date.today(), market_close) - datetime.datetime.combine(datetime.date.today(), current_time)).total_seconds()
        return time_to_close
    
    def get_positions(self):
        self.reqPositions()
    
    def submit_order(self, symbol, qty, side, order_type, secType = "STK", exchange = "SMART", currency = "USD"):
        contract = Contract()
        contract.symbol = symbol
        contract.secType = secType
        contract.exchange = exchange
        contract.currency = currency
        order = Order()
        order.action = side
        order.orderType = order_type
        order.totalQuantity = qty

        # Place the order
        self.nextOrderId = "ibkr_paper_" + uuid.uuid4()
        self.placeOrder(self.nextOrderId, contract, order)
    
class IBKRPaperTrading:
    def __init__(
        self,
        ticker_list,
        time_interval,
        drl_lib,
        agent,
        cwd,
        net_dim,
        state_dim,
        action_dim,
        API_KEY,
        API_SECRET,
        API_BASE_URL,
        tech_indicator_list,
        turbulence_thresh=30,
        max_stock=1e2,
        latency=None,
    ):
        # load agent
        self.drl_lib = drl_lib
        if agent == "ppo":
            if drl_lib == "elegantrl":
                from elegantrl.agents import AgentPPO
                from elegantrl.train.run import init_agent
                from elegantrl.train.config import (
                    Arguments,
                )  # bug fix:ModuleNotFoundError: No module named 'elegantrl.run'

                # load agent
                config = {
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                }
                args = Arguments(agent_class=AgentPPO, env=StockEnvEmpty(config))
                args.cwd = cwd
                args.net_dim = net_dim
                # load agent
                try:
                    agent = init_agent(args, gpu_id=0)
                    self.act = agent.act
                    self.device = agent.device
                except BaseException:
                    raise ValueError("Fail to load agent!")

            elif drl_lib == "rllib":
                from ray.rllib.agents import ppo
                from ray.rllib.agents.ppo.ppo import PPOTrainer

                config = ppo.DEFAULT_CONFIG.copy()
                config["env"] = StockEnvEmpty
                config["log_level"] = "WARN"
                config["env_config"] = {
                    "state_dim": state_dim,
                    "action_dim": action_dim,
                }
                trainer = PPOTrainer(env=StockEnvEmpty, config=config)
                trainer.restore(cwd)
                try:
                    trainer.restore(cwd)
                    self.agent = trainer
                    print("Restoring from checkpoint path", cwd)
                except:
                    raise ValueError("Fail to load agent!")

            elif drl_lib == "stable_baselines3":
                from stable_baselines3 import PPO

                try:
                    # load agent
                    self.model = PPO.load(cwd)
                    print("Successfully load model", cwd)
                except:
                    raise ValueError("Fail to load agent!")

            else:
                raise ValueError(
                    "The DRL library input is NOT supported yet. Please check your input."
                )

        else:
            raise ValueError("Agent input is NOT supported yet.")

        # connect to IBKR trading API
        try:
            self.ibkr_client = IBKRClient()
        except:
            raise ValueError(
                "Fail to connect Alpaca. Please check account info and internet connection."
            )

        # read trading time interval
        if time_interval == "1s":
            self.time_interval = 1
        elif time_interval == "5s":
            self.time_interval = 5
        elif time_interval == "1Min":
            self.time_interval = 60
        elif time_interval == "5Min":
            self.time_interval = 60 * 5
        elif time_interval == "15Min":
            self.time_interval = 60 * 15
        elif (
            time_interval == "1D"
        ):  # bug fix:1D ValueError: Time interval input is NOT supported yet. Maybe any other better ways
            self.time_interval = 24 * 60 * 60
        else:
            raise ValueError("Time interval input is NOT supported yet.")

        # read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_thresh = turbulence_thresh
        self.max_stock = max_stock

        # initialize account
        self.stocks = np.asarray([0] * len(ticker_list))  # stocks holding
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = None  # cash record
        self.stocks_df = pd.DataFrame(
            self.stocks, columns=["stocks"], index=ticker_list
        )
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))
        self.stockUniverse = ticker_list
        self.turbulence_bool = 0
        self.equities = []

    def test_latency(self, test_times=10):
        total_time = 0
        for i in range(0, test_times):
            time0 = time.time()
            self.get_state()
            time1 = time.time()
            temp_time = time1 - time0
            total_time += temp_time
        latency = total_time / test_times
        print("latency for data processing: ", latency)
        return latency

    def run(self):
        self.ibkr_client.reqOpenOrders()
        while not self.ibkr_client.open_orders_received:
            time.sleep(1)

        self.ibkr_client.cancel_open_orders()

        # Wait for market to open.
        print("Waiting for market to open...")
        tAMO = threading.Thread(target=self.awaitMarketOpen)
        tAMO.start()
        tAMO.join()
        print("Market opened.")
        while True:
            # Figure out when the market will close so we can prepare to sell beforehand.
            clossingTime = self.ibkr_client.calulateCloseTime()

            if clossingTime < (60):
                # Close all positions when 1 minutes til market close.
                print("Market closing soon. Stop trading.")
                self.close_positions()
            else:
                trade = threading.Thread(target=self.trade)
                trade.start()
                trade.join()
                last_equity = float(self.alpaca.get_account().last_equity)
                cur_time = time.time()
                self.equities.append([cur_time, last_equity])
                time.sleep(self.time_interval)

    def awaitMarketOpen(self):
        self.ibkr_client.reqCurrentTime()
        time.sleep(3)
        while not self.ibkr_client.is_market_open():
            self.ibkr_client.reqCurrentTime()
            time.sleep(1)  # Wait for the current time to be updated
            if self.ibkr_client.current_time:
                current_time = self.ibkr_client.current_time
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

    def trade(self):
        state = self.get_state()

        if self.drl_lib == "elegantrl":
            with torch.no_grad():
                s_tensor = torch.as_tensor((state,), device=self.device)
                a_tensor = self.act(s_tensor)
                action = a_tensor.detach().cpu().numpy()[0]

            action = (action * self.max_stock).astype(int)

        elif self.drl_lib == "rllib":
            action = self.agent.compute_single_action(state)

        elif self.drl_lib == "stable_baselines3":
            action = self.model.predict(state)[0]

        else:
            raise ValueError(
                "The DRL library input is NOT supported yet. Please check your input."
            )

        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            min_action = 10  # stock_cd
            for index in np.where(action < -min_action)[0]:  # sell_index:
                sell_num_shares = min(self.stocks[index], -action[index])
                qty = abs(int(sell_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(
                        qty, self.stockUniverse[index], "sell", respSO
                    )
                )
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0

            for index in np.where(action > min_action)[0]:  # buy_index:
                if self.cash < 0:
                    tmp_cash = 0
                else:
                    tmp_cash = self.cash
                buy_num_shares = min(
                    tmp_cash // self.price[index], abs(int(action[index]))
                )
                qty = abs(int(buy_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(
                        qty, self.stockUniverse[index], "buy", respSO
                    )
                )
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0

        else:  # sell all when turbulence
            positions = self.alpaca.list_positions()
            for position in positions:
                if position.side == "long":
                    orderSide = "sell"
                else:
                    orderSide = "buy"
                qty = abs(int(float(position.qty)))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(qty, position.symbol, orderSide, respSO)
                )
                tSubmitOrder.start()
                tSubmitOrder.join()

            self.stocks_cd[:] = 0

    def get_state(self):
        yfProcessor = YahooFinanceProcessor()
        price, tech, turbulence = yfProcessor.fetch_latest_data(
            ticker_list=self.stockUniverse,
            time_interval="1Min",
            tech_indicator_list=self.tech_indicator_list,
        )
        turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0

        turbulence = (
            self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2**-5
        ).astype(np.float32)

        tech = tech * 2**-7
        positions = self.alpaca.list_positions()
        stocks = [0] * len(self.stockUniverse)
        for position in positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = abs(int(float(position.qty)))

        stocks = np.asarray(stocks, dtype=float)
        cash = float(self.alpaca.get_account().cash)
        self.cash = cash
        self.stocks = stocks
        self.turbulence_bool = turbulence_bool
        self.price = price

        amount = np.array(self.cash * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        state = np.hstack(
            (
                amount,
                turbulence,
                self.turbulence_bool,
                price * scale,
                self.stocks * scale,
                self.stocks_cd,
                tech,
            )
        ).astype(np.float32)
        print(len(self.stockUniverse))
        return state

    def submitOrder(self, qty, stock, side, resp):
        if qty > 0:
            try:
                self.ibkr_client.submit_order(stock, qty, side, "MKT")
                print(
                    "Market order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | completed."
                )
                resp.append(True)
            except:
                print(
                    "Order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | did not go through."
                )
                resp.append(False)
        else:
            print(
                "Quantity is 0, order of | "
                + str(qty)
                + " "
                + stock
                + " "
                + side
                + " | not completed."
            )
            resp.append(True)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh

    def close_positions(self):
        print("Market closing soon. Closing positions.")
        self.ibkr_client.get_positions()
        time.sleep(5)  # Wait for positions to be fetched

        for contract, position, avgCost in self.positions:
            if position > 0:
                orderSide = 'SELL'
            else:
                orderSide = 'BUY'
            qty = abs(int(position))
            respSO = []
            tSubmitOrder = threading.Thread(target=self.submitOrder, args=(qty, contract.symbol, orderSide, respSO))
            tSubmitOrder.start()
            tSubmitOrder.join()

        # Run script again after market close for next trading day.
        print("Sleeping until market close (15 minutes).")
        time.sleep(60 * 15)

class StockEnvEmpty(gym.Env):
    # Empty Env used for loading rllib agent
    def __init__(self, config):
        state_dim = config["state_dim"]
        action_dim = config["action_dim"]
        self.env_num = 1
        self.max_step = 10000
        self.env_name = "StockEnvEmpty"
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.if_discrete = False
        self.target_return = 9999
        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )

    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        return

    def step(self, actions):
        return
