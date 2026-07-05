export LIBTORCH := env_var('HOME') + "/installs/libtorch"
export LD_LIBRARY_PATH := LIBTORCH + "/lib:" + env_var_or_default("LD_LIBRARY_PATH", "")
export LIBTORCH_CXX11_ABI := "0"

help:
    @just --list

format:
    cargo fmt

build: format
    cargo build

build_release: format
    cargo build --release

test: format
    cargo test -- --nocapture

default_ticker := "AMZN"

# Run any mode: `just run mode=run-rnn` or `just run mode=retrieve-data ticker=MSFT`
# Optional batch=32 enables shuffled mini-batch training for run-rnn/run-lstm.
run mode="retrieve-data" ticker=default_ticker batch="": build_release
    time cargo run --release -- --ticker {{ ticker }} --mode {{ mode }} {{ if batch != "" { "--batch-size " + batch } else { "" } }}

# Stage 1: download daily data, prep + scale windows, split, persist tensors
retrieve ticker=default_ticker: build_release
    time cargo run --release -- --ticker {{ ticker }} --mode retrieve-data

# Stage 2a: RNN on the persisted splits (batch="" -> full-batch)
train-rnn ticker=default_ticker batch="": build_release
    time cargo run --release -- --ticker {{ ticker }} --mode run-rnn {{ if batch != "" { "--batch-size " + batch } else { "" } }}

# Stage 2b: LSTM on the persisted splits (batch="" -> full-batch)
train-lstm ticker=default_ticker batch="": build_release
    time cargo run --release -- --ticker {{ ticker }} --mode run-lstm {{ if batch != "" { "--batch-size " + batch } else { "" } }}

## - just train-lstm → full-batch (unchanged)
##  - just train-lstm AMZN 32 ## does mini-batches of 32
##  - just train-rnn AMZN 64 ## RNN on AMZN with batch 64
##  - just run run-lstm AMZN 32 → generic runner with batching
