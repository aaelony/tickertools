use clap::{Parser, ValueEnum};

/// tickertools — retrieve ticker price history and train RNN/LSTM models.
#[derive(Parser, Debug)]
#[command(name = "tickertools", version, about)]
pub struct Cli {
    /// Stock ticker symbol to operate on (e.g. AMZN).
    #[arg(short, long, default_value = "AMZN")]
    pub ticker: String,

    /// Pipeline stage to run.
    #[arg(short, long, value_enum, default_value = "retrieve-data")]
    pub mode: RunMode,

    /// Mini-batch size for RunRnn/RunLstm.
    /// trains with shuffled mini-batches when set.
    /// uses full-batch gradient descent when omitted.
    #[arg(short, long)]
    pub batch_size: Option<i64>,
}

#[derive(ValueEnum, Debug, Clone, Copy)]
pub enum RunMode {
    /// Retrieve daily ticker price history data, prep to lookback, minmax scale to [-1,1], and save to ".ot" torch format.
    RetrieveData,
    /// Train a vanilla RNN on the persisted splits.
    RunRnn,
    /// Train an LSTM on the persisted splits.
    RunLstm,
}
