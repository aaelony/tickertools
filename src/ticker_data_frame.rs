use crate::price_row::PriceRow;
use chrono::{DateTime, Utc};
use std::fs::File;
use std::io::{BufWriter, Write};
use yfinance_rs::{Interval, Ticker, YfClient};

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

        // Had to tinker with this to get actual daily data not just first of the month days. Yahoo is wonky.
        let start = DateTime::<Utc>::from_timestamp(0, 0).expect("epoch is valid");
        let end = Utc::now();

        let price_history = ticker
            .history_builder()
            .between(start, end)
            .interval(Interval::D1)
            .fetch()
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
        writeln!(writer, "date{delimiter}close")?;
        for row in self.data.iter() {
            row.write_to_file(&mut writer, delimiter)?;
        }
        Ok(())
    }
}
