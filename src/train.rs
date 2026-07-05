use crate::split::TrainTestSplit;
use tch::{
    Device, Kind, Tensor, nn,
    nn::{OptimizerConfig, RNN},
};

#[derive(Debug, Clone, Copy)]
pub(crate) enum ModelKind {
    VanillaRnn,
    Lstm,
}

pub(crate) fn train(
    splits: &TrainTestSplit,
    device: Device,
    kind: ModelKind,
) -> Result<(), Box<dyn std::error::Error>> {
    tch::manual_seed(42);

    let x_train = splits.X_train.to_device(device).to_kind(Kind::Float);
    let y_train = splits.y_train.to_device(device).to_kind(Kind::Float);
    let x_test = splits.X_test.to_device(device).to_kind(Kind::Float);
    let y_test = splits.y_test.to_device(device).to_kind(Kind::Float);

    let n_train = x_train.size()[0];
    let seq_len = x_train.size()[1]; // 7 features -> 7 time steps
    let input_size: i64 = 1;
    let hidden: i64 = 32;
    let epochs: i64 = 5000;
    const TOL: f64 = 0.05; // tolerance for accuracy

    let accuracy = |yhat: &Tensor, y: &Tensor| -> f64 {
        let n = y.size()[0];
        let correct = (yhat - y).abs().lt(TOL).sum(Kind::Int64).int64_value(&[]);
        correct as f64 / n as f64
    };

    let vs = nn::VarStore::new(device);
    let root = &vs.root();
    let wy = nn::linear(root / "wy", hidden, 1, Default::default()); // shared output head

    //  the params are registered in `vs`
    let forward: Box<dyn Fn(&Tensor) -> Tensor> = match kind {
        ModelKind::VanillaRnn => {
            let wx = nn::linear(root / "wx", input_size, hidden, Default::default()); // W_x
            let wh = nn::linear(root / "wh", hidden, hidden, Default::default()); // W_h
            Box::new(move |x: &Tensor| {
                let batch = x.size()[0];
                let x_seq = x.reshape([batch, seq_len, input_size]); // (B, T, 1)
                let mut h = Tensor::zeros([batch, hidden], (Kind::Float, device)); // h_0 = 0
                for t in 0..seq_len {
                    let x_t = x_seq.narrow(1, t, 1).squeeze_dim(1); // (B, 1)
                    h = (x_t.apply(&wx) + h.apply(&wh)).tanh(); // h_t = tanh(x_t W_x + h_{t-1} W_h)
                }
                h.apply(&wy) // (B, 1)
            })
        }
        ModelKind::Lstm => {
            let lstm = nn::lstm(
                root / "lstm",
                input_size,
                hidden,
                nn::RNNConfig {
                    batch_first: true, // (B, T, I)
                    ..Default::default()
                },
            );
            Box::new(move |x: &Tensor| {
                let batch = x.size()[0];
                let x_seq = x.reshape([batch, seq_len, input_size]); // (B, T, 1)
                let (out, _state) = lstm.seq(&x_seq); // (B, T, H)
                let last = out.narrow(1, seq_len - 1, 1).squeeze_dim(1); // (B, H) at t0
                last.apply(&wy) // (B, 1)
            })
        }
    };

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    for epoch in 1..=epochs {
        let yhat = forward(&x_train);
        let loss = yhat.mse_loss(&y_train, tch::Reduction::Mean);
        opt.backward_step(&loss);

        if epoch == 1 || epoch % 10 == 0 {
            let (test_mse, test_acc) = tch::no_grad(|| {
                let yhat = forward(&x_test);
                let mse = yhat
                    .mse_loss(&y_test, tch::Reduction::Mean)
                    .double_value(&[]);

                (mse, accuracy(&yhat, &y_test))
            });

            println!(
                "Epoch {:4} | train MSE {:.6} | test MSE {:.6} | acc {:>5.1}%",
                epoch,
                loss.double_value(&[]),
                test_mse,
                test_acc * 100.0
            );
        }
    }

    println!("Trained {kind:?} on {n_train} windows (seq_len {seq_len}).");
    Ok(())
}

// Not for time-series, but other applications...
pub(crate) fn train_batched(
    splits: &TrainTestSplit,
    device: Device,
    kind: ModelKind,
    batch_size: i64,
) -> Result<(), Box<dyn std::error::Error>> {
    tch::manual_seed(42);

    // Move splits onto the training device as float32.
    let x_train = splits.X_train.to_device(device).to_kind(Kind::Float);
    let y_train = splits.y_train.to_device(device).to_kind(Kind::Float);
    let x_test = splits.X_test.to_device(device).to_kind(Kind::Float);
    let y_test = splits.y_test.to_device(device).to_kind(Kind::Float);

    let n_train = x_train.size()[0];
    let seq_len = x_train.size()[1]; // 7 features -> 7 time steps
    let input_size: i64 = 1;
    let hidden: i64 = 32;
    let epochs: i64 = 1000;
    const TOL: f64 = 0.05; // tolerance for accuracy

    let accuracy = |yhat: &Tensor, y: &Tensor| -> f64 {
        let n = y.size()[0];
        let correct = (yhat - y).abs().lt(TOL).sum(Kind::Int64).int64_value(&[]);
        correct as f64 / n as f64
    };

    let vs = nn::VarStore::new(device);
    let root = &vs.root();
    let wy = nn::linear(root / "wy", hidden, 1, Default::default()); // shared output head

    let forward: Box<dyn Fn(&Tensor) -> Tensor> = match kind {
        ModelKind::VanillaRnn => {
            let wx = nn::linear(root / "wx", input_size, hidden, Default::default()); // W_x
            let wh = nn::linear(root / "wh", hidden, hidden, Default::default()); // W_h
            Box::new(move |x: &Tensor| {
                let batch = x.size()[0];
                let x_seq = x.reshape([batch, seq_len, input_size]); // (B, T, 1)
                let mut h = Tensor::zeros([batch, hidden], (Kind::Float, device)); // h_0 = 0
                for t in 0..seq_len {
                    let x_t = x_seq.narrow(1, t, 1).squeeze_dim(1); // (B, 1)
                    h = (x_t.apply(&wx) + h.apply(&wh)).tanh(); // h_t = tanh(x_t W_x + h_{t-1} W_h)
                }
                h.apply(&wy) // (B, 1)
            })
        }
        ModelKind::Lstm => {
            let lstm = nn::lstm(
                root / "lstm",
                input_size,
                hidden,
                nn::RNNConfig {
                    batch_first: true, // (B, T, I)
                    ..Default::default()
                },
            );
            Box::new(move |x: &Tensor| {
                let batch = x.size()[0];
                let x_seq = x.reshape([batch, seq_len, input_size]); // (B, T, 1)
                let (out, _state) = lstm.seq(&x_seq); // (B, T, H)
                let last = out.narrow(1, seq_len - 1, 1).squeeze_dim(1); // (B, H) at t0
                last.apply(&wy) // (B, 1)
            })
        }
    };

    let mut opt = nn::Adam::default().build(&vs, 1e-3)?;

    for epoch in 1..=epochs {
        // Fresh shuffle each epoch — the mini-batch part.
        let perm = Tensor::randperm(n_train, (Kind::Int64, device));

        let mut epoch_loss = 0.0;
        let mut start = 0;
        let mut batch_idx_num = 0; // 0-based batch counter within the epoch
        while start < n_train {
            let len = batch_size.min(n_train - start); // last batch may be short
            let batch_idx = perm.narrow(0, start, len);
            let xb = x_train.index_select(0, &batch_idx); // (b, T)
            let yb = y_train.index_select(0, &batch_idx); // (b, 1)

            let yhat = forward(&xb);
            let loss = yhat.mse_loss(&yb, tch::Reduction::Mean);
            opt.backward_step(&loss);

            let batch_loss = loss.double_value(&[]);
            if batch_idx_num % 100 == 0 {
                let batch_acc = accuracy(&yhat, &yb);
                println!(
                    "Epoch {:4} | batch {:4} | batch MSE {:.6} | batch acc {:>5.1}%",
                    epoch,
                    batch_idx_num,
                    batch_loss,
                    batch_acc * 100.0
                );
            }

            epoch_loss += batch_loss * len as f64; // weight by batch size
            start += batch_size;
            batch_idx_num += 1;
        }
        let train_mse = epoch_loss / n_train as f64;

        if epoch == 1 || epoch % 10 == 0 {
            let (test_mse, test_acc) = tch::no_grad(|| {
                let yhat = forward(&x_test);
                let mse = yhat
                    .mse_loss(&y_test, tch::Reduction::Mean)
                    .double_value(&[]);

                (mse, accuracy(&yhat, &y_test))
            });
            println!(
                "Epoch {:4} | train MSE {:.6} | test MSE {:.6} | test acc {:>5.1}%",
                epoch,
                train_mse,
                test_mse,
                test_acc * 100.0
            );
        }
    }

    println!("Trained {kind:?} on {n_train} windows (seq_len {seq_len}, batch {batch_size}).");
    Ok(())
}
