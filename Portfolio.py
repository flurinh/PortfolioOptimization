import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import yfinance as yf
TICKER_LIST = ['ES=F', 'MSFT']


# Todo: Following is the class 'States' which we can use to get real-time stock-data from yahoo
# Todo: As per suggestion of other group members we could also implement a function yielding artificial data to fit to
# Todo: first, since that may be easier to evaluate!


def refresh_input(ticker_list=TICKER_LIST, period='1d', interval='1m'):
    # Todo: Implement function to update input data
    states = {}
    for tick in ticker_list:
        ticker = yf.Ticker(tick)
        hist = ticker.history(period=period, interval=interval)
        dates = hist.index.values
        start = dates[0]
        steps = dates - start
        High = hist["High"].to_numpy()
        Low = hist['Low'].to_numpy()
        additional_info = ''  # Todo: Not implemented, maybe some risk-assessment
        states.update({'time': steps, tick: {'history': {'High': High, 'Low': Low}, 'info': additional_info}})
    return states


class States:
    def __init__(self,
                 idx = 0,
                 static=True,
                 dynamic=False,
                 refresh_rate=1,
                 plot_states=True):
        self.static = static
        self.dynamic = dynamic
        self.refresh_rate = refresh_rate  # how often are stats refreshed in dynamic setting
        self.period = None
        self.interval = None
        self.set_time()
        self.tick = None
        self.set_tick(idx)  # default initialization

        # Next some visualization, NOTE that dynamic analysis is more of a real-time visualization method so far rather
        # than a function to update values in real-time and predicting for them (there is an infinite while loop ...)
        self.states = None
        self.plot_states = plot_states
        if self.static:
            print("Running static analysis!")
            self.static_analysis()
        if self.dynamic:
            print("Running dynamic analysis!")
            self.dynamic_analysis()

    def set_tick(self, idx):
        self.tick = TICKER_LIST[idx]
        print("Current ticker is ", self.tick)

    def set_time(self, period='1d', interval='1m'):
        self.period = period
        self.interval = interval
        print('Current analysis time period is {} and interval is {}.'.format(self.period, self.interval))

    def dynamic_analysis(self):
        # dynamic analysis
        while True:
            self.states = refresh_input(ticker_list=TICKER_LIST, period='1d', interval='1m')
            if self.plot_states:
                self.visualize_dynamic_states()
            time.sleep(self.refresh_rate)

    def static_analysis(self):
        # static analysis
        self.states = refresh_input()
        if self.plot_states:
            self.visualize_static_states()

    def return_states(self):
        # return all data used by the optimization class ---------------> THIS IS THE FUNCTION WE WILL CALL TO GET DATA!
        return self.states

    def visualize_dynamic_states(self, show_idx=0):
        # Todo: This function works, but should be optimized to update the plot-canvis rather than create a new one.
        tick = TICKER_LIST[show_idx]
        df = self.states[tick]
        high = df['history']['High']
        low = df['history']['Low']
        length = low.shape[0]
        t = self.states['time'][:length].astype(float)
        high = high[:t.shape[0]]
        low = low[:t.shape[0]]
        plt.plot(t, high)
        plt.plot(t, low)
        plt.show()
        plt.pause(self.refresh_rate)

    def visualize_static_states(self, show_idx=None):
        if show_idx != None:
            self.set_tick(show_idx)
        df = self.states[self.tick]
        high = df['history']['High']
        low = df['history']['Low']
        length = low.shape[0]
        t = self.states['time'][:length].astype(float)
        high = high[:t.shape[0]]
        low = low[:t.shape[0]]
        plt.plot(t, high)
        plt.plot(t, low)
        plt.show()