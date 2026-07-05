// use chrono::{DateTime, Utc};
//use ndarray::Array1;
pub mod price_row;
pub mod ticker_data_frame;
// pub mod ticker_lookback_df;

use tch::{
    Device, Kind, Tensor, nn,
    nn::{OptimizerConfig, RNN},
};
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

// Batches of Random Integer data
fn make_batch(batch: i64, t_steps: i64, vocab: i64, device: Device) -> (Tensor, Tensor, Tensor) {
    let x_idx = Tensor::randint(vocab, [batch, t_steps], (Kind::Int64, device));
    let y_idx = (&x_idx + 1).remainder(vocab);

    // teacher : shift by 1 (mod V)
    let x_onehot = x_idx.one_hot(vocab).to_kind(Kind::Float);

    println!(
        "New Batch info:\n\tx_idx: {:?}\n\n\ty_idx: {:?}\n\n\tx_onehot: {:?}\n",
        &x_idx, &y_idx, &x_onehot,
    );

    // (B, T, V)
    (x_idx, y_idx, x_onehot)
}

pub fn run_rnn() -> Result<(), Box<dyn std::error::Error>> {
    tch::manual_seed(42);

    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    let root = &vs.root();

    let vocab: i64 = 6;
    let hidden: i64 = 16;
    let t_steps: i64 = 8;
    let batch: i64 = 32;
    let epochs: i64 = 120;

    // RNN
    let wx = nn::linear(root / "wx", vocab, hidden, Default::default()); // W_x
    let wh = nn::linear(root / "wh", hidden, hidden, Default::default()); // W_h
    let wy = nn::linear(root / "wy", hidden, vocab, Default::default()); // W_y

    // LSTM
    let lstm = nn::lstm(
        root / "lstm",
        vocab,
        hidden,
        nn::RNNConfig {
            num_layers: 1,
            bidirectional: false,
            batch_first: true, // (B, T, I)
            ..Default::default()
        },
    );

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?; // Optimizer

    for epoch in 1..=epochs {
        let (_x_idx, y_idx, x_oh) = make_batch(batch, t_steps, vocab, device);

        let mut h = Tensor::zeros([batch, hidden], (Kind::Float, device)); // h_0 = 0 in (B,H)
        let mut logits_per_t: Vec<Tensor> = Vec::with_capacity(t_steps as usize); // Collect (B, V) logits for each time step

        for t in 0..t_steps {
            // Vanilla RNN start
            let x_t = x_oh.narrow(1, t, 1).squeeze_dim(1); // slice the t-th input (B, 1, V) -> (B, V)

            let a = x_t.apply(&wx) + h.apply(&wh); // pre-activation: a_t = x_t@W_x  +  h_{t-1}@W_h
            h = a.tanh(); // hidden update: h_t = tanh(a_t)

            let logits_t = h.apply(&wy); //  (B, V)
            // Vanilla RNN end

            logits_per_t.push(logits_t);
        }

        // ---------------------------
        // LSTM Start
        let (h_seq, _state) = lstm.seq(&x_oh); // (B, T, H)

        // LSTM end

        let logits = Tensor::stack(&logits_per_t, 1); // stack back to (B, T, V) for VanillaRNN
        let logits = h_seq.apply(&wy); // (B, T, V) for LSTM

        let loss = logits
            .reshape([batch * t_steps, vocab])
            .cross_entropy_for_logits(&y_idx.reshape([batch * t_steps])); // (B*T)

        opt.backward_step(&loss); // build the graph across time and runs Backpropagation Through Time (BPTT)

        let (_x_eval_idx, y_eval_idx, x_eval_oh) = make_batch(1, t_steps, vocab, device);
        let mut h_eval = Tensor::zeros([1, hidden], (Kind::Float, device));
        let mut eval_logits_per_t: Vec<Tensor> = Vec::with_capacity(t_steps as usize);

        for t in 0..t_steps {
            let x_t = x_eval_oh.narrow(1, t, 1).squeeze_dim(1); // (1, V)
            h_eval = (x_t.apply(&wx) + h_eval.apply(&wh)).tanh();
            eval_logits_per_t.push(h_eval.apply(&wy));
        }
        let logits_eval = Tensor::stack(&eval_logits_per_t, 1); // (1, T, V)
        let preds = logits_eval.argmax(-1, false); // (1, T)

        // Convert Tensors to vectors and compute accuracy
        let preds_vec: Vec<i64> = preds
            .to_device(Device::Cpu)
            .view([-1])
            .iter::<i64>()?
            .collect();

        let y_vec: Vec<i64> = y_eval_idx
            .to_device(Device::Cpu)
            .view([-1])
            .iter::<i64>()?
            .collect();

        let correct = preds_vec
            .iter()
            .zip(y_vec.iter())
            .filter(|(a, b)| a == b)
            .count();

        let accuracy = correct as f64 / preds_vec.len() as f64;

        let loss_val = loss.to_device(Device::Cpu).double_value(&[]);

        println!(
            "Epoch {:3} | Loss {:4} | Eval Accuracy {:>5.1}%",
            epoch,
            loss_val,
            accuracy * 100.0
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Tensor;

    #[test]
    fn test_make_batch() {
        let batch_size = 10_i64;
        let t_steps = 7_i64;
        let vocab = 5_i64;
        let device = Device::cuda_if_available();

        let (x_idx, y_idx, x_onehot) = make_batch(batch_size, t_steps, vocab, device);
        println!(
            "batch_size: {}, t_steps: {}, vocab: {}, device: {:?}",
            batch_size, t_steps, vocab, device
        );
        println!(
            "Output: x_idx: {}, y_idx: {}, x_onehot: {}",
            x_idx, y_idx, x_onehot
        );
    }
}
