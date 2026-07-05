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

pub fn minmax_scale_tensor(
    data: &Tensor,
    feature_min: f64,
    feature_max: f64,
) -> (Tensor, Tensor, Tensor) {
    let min = data.min();
    let max = data.max();

    let range = &max - &min;
    let range = range.clamp_min(1e-12);
    let scaled_0to1 = (data - &min) / &range;
    let feat_range = feature_max - feature_min;
    (scaled_0to1 * feat_range + feature_min, min, max)
}

pub fn minmax_unscale_tensor(
    scaled: &Tensor,
    min: &Tensor,
    max: &Tensor,
    feature_min: f64,
    feature_max: f64,
) -> Tensor {
    let range = (max - min).clamp_min(1e-12);
    let feat_range = feature_max - feature_min;

    // invert: scaled = ((data - min) / range) * feat_range + feature_min
    let scaled_0to1 = (scaled - feature_min) / feat_range;
    scaled_0to1 * &range + min
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

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_minmax_roundtrip() {
        let orig = Tensor::from_slice(&[2.0f64, 4.0, 6.0, 8.0, 10.0]);
        let feature_min = -1.0;
        let feature_max = 1.0;
        let (scaled, min, max) = minmax_scale_tensor(&orig, feature_min, feature_max);

        let scaled_min: f64 = scaled.min().double_value(&[]);
        let scaled_max: f64 = scaled.max().double_value(&[]);
        assert!(scaled_min >= feature_min - 1e-6);
        assert!(scaled_max <= feature_max + 1e-6);

        assert!((min.double_value(&[]) - 2.0).abs() < 1e-6);
        assert!((max.double_value(&[]) - 10.0).abs() < 1e-6);

        let unscaled = minmax_unscale_tensor(&scaled, &min, &max, feature_min, feature_max);

        println!("\n\tOrig Tensor: {:?}", orig);

        println!(
            "\tDetermined min({}) and\n\tmax({})",
            min.double_value(&[]),
            max.double_value(&[]),
        );
        println!(
            "\tScaled Tensor from feature_min ({feature_min}) to feature_max ({feature_max}): {:?}",
            scaled
        );

        println!("\tUnscaled (recovered) Tensor: {:?}", unscaled);

        // Compare recovered vs original, element-wise, within tolerance
        let diff = (&unscaled - &orig).abs();
        let max_diff: f64 = diff.max().double_value(&[]);
        assert!(max_diff < 1e-6, "round-trip error too large: {}", max_diff);
    }
}
