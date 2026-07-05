# Ticker Tools

A toy rust project using [tch-rs](https://github.com/laurentmazare/tch-rs) that:

- Has a Run mode to retrieve data:
  - Downloads historical daily stock prices for a ticker.
  - Prepares a  7-day lookback version of the daily data with columns `date	tm7	tm6	tm5	tm4	tm3	tm2	tm1	t0`, where `tm7` (time minus 7) means 7 days from the current `t0`.
  - MinMax scales the 7-day lookback over `[-1, 1]`.
- Has a Run mode to train an RNN to predict the daily  price from the previous 7 days.
- Has a Run mode to train an LSTM ...
