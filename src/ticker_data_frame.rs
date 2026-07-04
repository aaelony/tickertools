//use chrono::{DateTime, Utc};
//use ndarray::Array1;
use crate::price_row::PriceRow;
use chrono::{DateTime, Utc};
// use ndarray::Array1;
use std::io::BufWriter;
use std::{f64, fs::File};
use tch::Tensor;
use yfinance_rs::{Interval, Range, Ticker, YfClient};

#[derive(Debug, Clone)]
pub struct TickerDataFrame {
    pub ticker_name: String,
    pub data: Vec<PriceRow>,
}

impl TickerDataFrame {
    pub async fn new(ticker_name: String) -> Result<Self, Box<dyn std::error::Error>> {
        Self::retrieve_data(ticker_name).await
    }

    // Use the Yahoo Finance Client to pull closing price history for a stock ticker
    // Return a Result<TickerDataFrame, Error>
    pub async fn retrieve_data(ticker_name: String) -> Result<Self, Box<dyn std::error::Error>> {
        let client = YfClient::default();
        let ticker = Ticker::new(&client, &ticker_name);

        let price_history = ticker
            .history(Some(Range::Max), Some(Interval::D1), true)
            .await?;

        println!("row cnt: {}", price_history.len());

        let data: Vec<PriceRow> = price_history
            .into_iter()
            .map(|row| PriceRow {
                date: row.ts,
                close: row.ohlc.close.as_decimal().as_f64(),
            })
            .collect();

        Ok(TickerDataFrame { ticker_name, data })
    }

    pub async fn write_to_file(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut writer = BufWriter::new(File::create(format!("{}_daily.tsv", self.ticker_name))?);
        let delimiter = "\t";
        for row in self.data.iter() {
            row.write_to_file(&mut writer, delimiter)?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct WindowRow {
    pub date: DateTime<Utc>,
    //  pub closes: [f64; 8], // [-7, -6, ..., -1, today]
    // pub closes: Array1<f64>, // length = 8
    pub closes: Tensor,
}

/// make_ts_windows(df.clone(), 8)
// 8 = 7 features + 1 outcome
pub fn make_ts_windows(df: TickerDataFrame, lookback: usize) -> Vec<WindowRow> {
    df.data
        .windows(lookback)
        .map(|w| {
            let values: Vec<f64> = w.iter().map(|row| row.close).collect();

            WindowRow {
                date: w[0].date,
                // closes: Array1::from_vec(w.iter().map(|row| row.close).collect()),
                closes: Tensor::from_slice(&values),
            }
        })
        .collect()
}

#[derive(Debug)]
pub struct PreppedDataFrame {
    pub data: Vec<WindowRow>,
}

impl PreppedDataFrame {
    pub fn new(ticker_price_history_df: TickerDataFrame, lookback: usize) -> Self {
        let data: Vec<WindowRow> = make_ts_windows(ticker_price_history_df, lookback);
        Self { data }
    }

    pub fn minmax_scale(data: &[WindowRow], feature_min: f64, feature_max: f64) -> Vec<WindowRow> {
        let mut all_min = f64::INFINITY;
        let mut all_max = f64::NEG_INFINITY;

        for w in data {
            for val in w.closes.to_kind(tch::Kind::Double).iter::<f64>().unwrap() {
                let fv = val as f64;
                if fv < all_min {
                    all_min = fv;
                }
                if fv > all_max {
                    all_max = fv;
                }
            }
        }

        let input_range = all_max - all_min;
        let output_range = feature_max - feature_min;

        data.iter()
            .map(|w| {
                let scaled_closes =
                    (&w.closes - all_min) / input_range * output_range + feature_min;
                WindowRow {
                    date: w.date,
                    closes: scaled_closes,
                }
            })
            .collect()
    }
}
