pub mod cli;
pub mod prep;
pub mod price_row;
mod scratch;
pub mod split;
pub mod ticker_data_frame;
pub mod train;

use clap::Parser;
use cli::{Cli, RunMode};
use prep::PreppedDataFrame;
use split::TrainTestSplit;
use tch::Device;
use train::{ModelKind, train, train_batched};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let ticker_name = cli.ticker;
    let batch_size = cli.batch_size;

    match cli.mode {
        RunMode::RetrieveData => {
            let df_price_history =
                ticker_data_frame::TickerDataFrame::new(ticker_name.clone()).await?;
            println!("This is df_price_history: {:?}", df_price_history);
            df_price_history.write_to_file().await?;

            let df_prepped = PreppedDataFrame::new(df_price_history, 8);

            let df_scaled_prepped = PreppedDataFrame {
                data: PreppedDataFrame::minmax_scale(&df_prepped.data, -1.0, 1.0),
            };

            println!("This is df_prepped: {:?}", df_prepped);
            df_prepped.write_to_file(&ticker_name)?;

            println!("\n\n\nThis is df_scaled_prepped: {:?}", df_scaled_prepped);
            df_scaled_prepped.write_scaled_to_file(&ticker_name, -1.0, 1.0)?;

            let split_pct = 0.95;

            let splits = df_scaled_prepped.train_test_split(split_pct);
            println!(
                "Split sizes -> X_train: {:?}, X_test: {:?}, y_train: {:?}, y_test: {:?}",
                splits.X_train.size(),
                splits.X_test.size(),
                splits.y_train.size(),
                splits.y_test.size(),
            );
            splits.save(format!("{ticker_name}_splits.ot"))?;

            //println!("X_train: {:?}\n\n", splits.X_train);
            //println!("X_test: {:?}\n\n", splits.X_test);
        }
        RunMode::RunRnn => {
            let device = Device::cuda_if_available();
            println!("Training RNN on device: {:?}", device);

            let splits = TrainTestSplit::load(format!("{ticker_name}_splits.ot"))?;
            match batch_size {
                Some(bs) => train_batched(&splits, device, ModelKind::VanillaRnn, bs)?, // batching doesn't make sense for Time Series prediction, but just to try.
                None => train(&splits, device, ModelKind::VanillaRnn)?,
            }
        }
        RunMode::RunLstm => {
            let device = Device::cuda_if_available();
            println!("Training LSTM on device: {:?}", device);

            let splits = TrainTestSplit::load(format!("{ticker_name}_splits.ot"))?;
            match batch_size {
                Some(bs) => train_batched(&splits, device, ModelKind::Lstm, bs)?,
                None => train(&splits, device, ModelKind::Lstm)?,
            }
        }
    }

    Ok(())
}
