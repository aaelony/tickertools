use crate::split::TrainTestSplit;
use crate::ticker_data_frame::TickerDataFrame;
use chrono::{DateTime, Utc};
use std::fs::File;
use std::io::{BufWriter, Write};
use tch::IndexOp;
use tch::Tensor;

#[derive(Debug)]
pub struct WindowRow {
    pub date: DateTime<Utc>,
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
                date: w[lookback - 1].date, // t0: most recent day in the window
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
    pub data: Vec<WindowRow>, // WindowRow contains date and closes tensor
}

impl PreppedDataFrame {
    pub fn new(ticker_price_history_df: TickerDataFrame, lookback: usize) -> Self {
        let data: Vec<WindowRow> = make_ts_windows(ticker_price_history_df, lookback);
        Self { data }
    }

    /// price history tensor partitioned into: features X (tm7..tm1) and target y (t0),
    /// then into train/test by `train_frac` (e.g. 0.95 => 95% train, 5% test).
    pub fn train_test_split(&self, train_frac: f64) -> TrainTestSplit {
        // Stack each window's closes into an [N, lookback] matrix.
        let rows: Vec<Tensor> = self.data.iter().map(|w| w.closes.shallow_clone()).collect();
        let matrix = Tensor::stack(&rows, 0);

        let n = matrix.size()[0];
        let lookback = matrix.size()[1];

        // X = all columns except the last (tm7..tm1); y = the last column (t0).
        let X = matrix.narrow(1, 0, lookback - 1); // [N, lookback - 1]
        let y = matrix.narrow(1, lookback - 1, 1); // [N, 1]

        let split_index = (n as f64 * train_frac) as i64;

        println!(
            "The last row of X_train is row {}, X: {}, y: {}",
            (split_index + 2), // Real data has a header row and we start counting at row 1, so add 1 + 1 = 2 to split_index.
            X.i(split_index),
            y.i(split_index),
        );

        TrainTestSplit {
            X_train: X.narrow(0, 0, split_index), // https://docs.pytorch.org/docs/2.12/generated/torch.narrow.html.  args: dim, start, length
            X_test: X.narrow(0, split_index, n - split_index),
            y_train: y.narrow(0, 0, split_index),
            y_test: y.narrow(0, split_index, n - split_index),
        }
    }

    fn window_header(&self, delimiter: &str) -> String {
        let n = self.data.first().map_or(0, |w| w.closes.size()[0]);
        let mut header = String::from("date");
        for i in 0..n {
            let label = if i == n - 1 {
                "t0".to_string()
            } else {
                format!("tm{}", n - 1 - i)
            };
            header.push_str(delimiter);
            header.push_str(&label);
        }
        header
    }

    pub fn write_to_file(&self, ticker_name: &str) -> Result<(), Box<dyn std::error::Error>> {
        let mut writer = BufWriter::new(File::create(format!("{ticker_name}_daily_lookback.tsv"))?);
        let delimiter = "\t";
        writeln!(writer, "{}", self.window_header(delimiter))?;
        for wr in &self.data {
            write!(writer, "{}", wr.date.format("%Y-%m-%d"))?;
            for close in wr.closes.to_kind(tch::Kind::Double).iter::<f64>()? {
                write!(writer, "{delimiter}{close}")?;
            }
            writeln!(writer)?;
        }
        Ok(())
    }

    pub fn write_scaled_to_file(
        &self,
        ticker_name: &str,
        feature_min: f64,
        feature_max: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut writer = BufWriter::new(File::create(format!(
            "{ticker_name}_daily_lookback_scaled_{feature_min}_to_{feature_max}.tsv"
        ))?);
        let delimiter = "\t";
        writeln!(writer, "{}", self.window_header(delimiter))?;
        for wr in &self.data {
            write!(writer, "{}", wr.date.format("%Y-%m-%d"))?;
            for close in wr.closes.to_kind(tch::Kind::Double).iter::<f64>()? {
                write!(writer, "{delimiter}{close}")?;
            }
            writeln!(writer)?;
        }
        Ok(())
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
