use chrono::{DateTime, Duration, Utc};
use ndarray::Array1;
use std::fs::File;
use std::io::{BufWriter, Write};
use yfinance_rs::{Candle, Interval, Range, Ticker, YfClient};

#[derive(Debug, Clone)]
pub struct WindowRow {
    pub date: DateTime<Utc>,
    //  pub closes: [f64; 8], // [-7, -6, ..., -1, today]
    pub closes: Array1<f64>, // length = 8
}

pub fn make_windows(candles: &[Candle]) -> Vec<WindowRow> {
    candles
        .windows(8)
        .map(|w| WindowRow {
            date: w[7].ts,
            closes: Array1::from_vec(vec![
                w[0].ohlc.close.as_decimal().as_f64(), // tm7 = t minus 7 days
                w[1].ohlc.close.as_decimal().as_f64(), // tm6
                w[2].ohlc.close.as_decimal().as_f64(), // tm5
                w[3].ohlc.close.as_decimal().as_f64(), // tm4
                w[4].ohlc.close.as_decimal().as_f64(), // tm3
                w[5].ohlc.close.as_decimal().as_f64(), // tm2
                w[6].ohlc.close.as_decimal().as_f64(), // previous day, tm1
                w[7].ohlc.close.as_decimal().as_f64(), // most recent day tm0
            ]),
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct PriceRow {
    pub date: Utc,
    pub close: f64,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ticker_str = String::from("AMZN");
    let output_delimiter = String::from("\t");
    let client = YfClient::default();
    let ticker = Ticker::new(&client, ticker_str.clone());

    let price_history = ticker
        .history(Some(Range::Max), Some(Interval::D1), true)
        .await?;

    // dbg!(std::any::type_name_of_val(&price_history));
    // dbg!(&price_history[0]);

    let cutoff = Utc::now() - Duration::days(23 * 365);

    let mut writer = BufWriter::new(File::create(format!("{}_daily.tsv", ticker_str))?);
    writeln!(
        writer,
        "{}",
        format!("date{}close", output_delimiter.clone())
    )?;

    let df = make_windows(&price_history);
    println!("{:?}", df);

    for bar in price_history.clone() {
        if bar.ts >= cutoff {
            writeln!(
                writer,
                "{}{}{}",
                bar.ts.format("%Y-%m-%d"),
                output_delimiter,
                bar.ohlc.close
            )?;
        }
    }

    Ok(())
}
