use ndarray::{Array, ArrayBase, Dim, OwnedRepr};
use rand::Rng;
use tch::nn::OptimizerConfig;
use tch::{nn, no_grad, Kind, Tensor};

fn model(p: &nn::Path) -> impl nn::Module {
    nn::seq()
        .add(nn::linear(p / "lin1", 1, 32, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "lin3", 32, 16, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(p / "lin2", 16, 1, Default::default()))
}

fn main() {
    let mut rng = rand::thread_rng();
    let x = Array::range(0., 3., 0.01);
    let w = 1.;
    let b = -2.;
    let d = 3.;
    let mut noise = vec![];
    for _ in 0..300 {
        noise.push(rng.gen_range(0.0..1.0));
    }
    let noise: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = ArrayBase::from_vec(noise);
    let y: ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> = w * &x * &x + b * &x + d;
    let y = y + noise;

    let x = Tensor::of_slice(&x.to_vec())
        .to_kind(Kind::Float)
        .unsqueeze(-1);
    // x.print();
    let y = Tensor::of_slice(&y.to_vec())
        .to_kind(Kind::Float)
        .unsqueeze(-1);

    let vs = nn::VarStore::new(tch::Device::Cpu);
    let model = model(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-2);

    let y_test = no_grad(|| x.apply(&model));
    let pre_test = Tensor::cat(&[&y, &y_test], 1);
    pre_test.print();

    for i in 0..2000 {
        let y_pre = x.apply(&model);
        // y_pre.print();
        let loss = (&y - y_pre).pow(&Tensor::from(2.0)).mean(Kind::Float);
        if i % 200 == 0 {
            loss.print();
        }

        match opt {
            Ok(ref mut opt) => {
                opt.backward_step(&loss);
            }
            Err(_) => {}
        }
    }
    let y_pre = x.apply(&model);
    let ans = Tensor::cat(&[y, y_pre], 1);
    ans.print();
}
