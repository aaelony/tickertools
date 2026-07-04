// use chrono::{DateTime, Utc};
//use ndarray::Array1;
pub mod price_row;
pub mod ticker_data_frame;
// pub mod ticker_lookback_df;

use ticker_data_frame::PreppedDataFrame;

#[derive(Debug)]
enum RunMode {
    RetrieveData(String),
    RunRnn(String),
    RunLSTM(String),
}

pub fn run_vanilla_rnn() {}

pub fn run_lstm() {}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ticker_name = String::from("AMZN");

    let run_mode = RunMode::RetrieveData(ticker_name);

    match run_mode {
        RunMode::RetrieveData(ticker_name) => {
            let df_price_history = ticker_data_frame::TickerDataFrame::new(ticker_name).await?;
            println!("This is df_price_history: {:?}", df_price_history);

            let df_prepped = PreppedDataFrame::new(df_price_history, 8);

            let df_scaled_prepped = PreppedDataFrame::minmax_scale(&df_prepped.data, -1.0, 1.0);

            println!("This is df_prepped: {:?}", df_prepped);
            println!("\n\n\nThis is df_scaled_prepped: {:?}", df_scaled_prepped);
        }
        RunMode::RunRnn(_ticker_name) => {}
        RunMode::RunLSTM(_ticker_name) => {}
    }

    Ok(())
}
