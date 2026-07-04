use chrono::{DateTime, Utc};

use std::fs::File;
use std::io::{BufWriter, Write};

#[derive(Debug, Clone)]
pub struct PriceRow {
    pub date: DateTime<Utc>,
    pub close: f64,
}

impl PriceRow {
    pub fn write_to_file(
        &self,
        writer: &mut BufWriter<File>,
        delimiter: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        writeln!(
            writer,
            "{}{}{}",
            self.date.format("%Y-%m-%d"),
            delimiter,
            self.close
        )?;
        Ok(())
    }
}
