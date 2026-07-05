use std::collections::HashMap;
use std::path::Path;
use tch::Tensor;

#[derive(Debug)]
#[allow(non_snake_case)]
pub struct TrainTestSplit {
    pub X_train: Tensor,
    pub X_test: Tensor,
    pub y_train: Tensor,
    pub y_test: Tensor,
}

impl TrainTestSplit {
    // https://docs.rs/tch/0.19.0/src/tch/wrappers/tensor.rs.html#631-640
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        Tensor::save_multi(
            &[
                ("X_train", &self.X_train),
                ("X_test", &self.X_test),
                ("y_train", &self.y_train),
                ("y_test", &self.y_test),
            ],
            path,
        )?;
        Ok(())
    }

    // https://docs.rs/tch/0.19.0/src/tch/wrappers/tensor.rs.html#631-640
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let mut map: HashMap<String, Tensor> = Tensor::load_multi(path)?.into_iter().collect();
        let take = |m: &mut HashMap<String, Tensor>, k: &str| {
            m.remove(k)
                .ok_or_else(|| format!("missing tensor `{k}` in split file"))
        };
        Ok(Self {
            X_train: take(&mut map, "X_train")?,
            X_test: take(&mut map, "X_test")?,
            y_train: take(&mut map, "y_train")?,
            y_test: take(&mut map, "y_test")?,
        })
    }
}
