# Ticker Tools

A toy rust project using [tch-rs](https://github.com/laurentmazare/tch-rs) that:

- Has a Run mode to retrieve data:
  - Downloads historical daily stock prices for a ticker.
  - Prepares a  7-day lookback version of the daily data with columns `date	tm7	tm6	tm5	tm4	tm3	tm2	tm1	t0`, where `tm7` (time minus 7) means 7 days from the current `t0`.
  - MinMax scales the 7-day lookback over `[-1, 1]`.
- Has a Run mode to train an RNN to predict the daily  price from the previous 7 days.
- Has a Run mode to train an LSTM ...



```
just build_release
just_retrieve AMZN
just train-lstm AMZN
```

```
    Finished `release` profile [optimized] target(s) in 0.07s
     Running `target/release/tickertools --ticker AMZN --mode run-lstm`
Training LSTM on device: Cpu
Epoch    1 | train MSE 0.409503 | test MSE 0.245938 | acc   2.2%
Epoch   10 | train MSE 0.144890 | test MSE 0.191327 | acc   2.5%
Epoch   20 | train MSE 0.028738 | test MSE 0.150373 | acc   3.8%
Epoch   30 | train MSE 0.017358 | test MSE 0.127651 | acc   4.1%
Epoch   40 | train MSE 0.010655 | test MSE 0.116432 | acc   3.5%
Epoch   50 | train MSE 0.004703 | test MSE 0.102271 | acc   3.5%
Epoch   60 | train MSE 0.004064 | test MSE 0.080052 | acc   6.8%
Epoch   70 | train MSE 0.003155 | test MSE 0.058698 | acc  14.4%
Epoch   80 | train MSE 0.002635 | test MSE 0.045011 | acc  23.7%

...

Epoch 4970 | train MSE 0.000107 | test MSE 0.003045 | acc  78.5%
Epoch 4980 | train MSE 0.000107 | test MSE 0.003048 | acc  78.5%
Epoch 4990 | train MSE 0.000107 | test MSE 0.003052 | acc  78.5%
Epoch 5000 | train MSE 0.000107 | test MSE 0.003056 | acc  78.5%
Trained Lstm on 6957 windows (seq_len 7).
```
