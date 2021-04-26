pub mod math;
pub mod dnn;
pub mod mnist;

use std::f64::NAN;
pub use math::matrix;
pub use math::distribution;
//pub use neuralnetwork::dnn::neuralNetwork;
pub use ndarray::{arr2, Array2};
// pub use mnist::mnistUtil;
use mnist::mnistUtil;

#[cfg(test)]
mod tests {
    
   // use super::sorting;
    use super::*;
    fn testMax(){
        let a = arr2(&[[0.],[0.],[0.],[1.],[0.],[0.],[0.],[0.],[0.],[0.]]);
        let b = dnn::neuralNetwork::max(&a);
        assert_eq!(3,b);
        assert_eq!(NAN, 0.0_f64.ln());
    }

    fn testMax2(){
     let train_image = mnistUtil::readFeaturesFromFile("D:\\rust-pj\\neuralnetwork\\src\\mnist\\train-images.idx3-ubyte");
     let train_label = mnistUtil::readLabelsFromFile("D:\\rust-pj\\neuralnetwork\\src\\mnist\\train-labels.idx1-ubyte");

     let test_image = mnistUtil::readFeaturesFromFile("D:\\rust-pj\\neuralnetwork\\src\\mnist\\t10k-images.idx3-ubyte");
     let test_label = mnistUtil::readLabelsFromFile("D:\\rust-pj\\neuralnetwork\\src\\mnist\\t10k-labels.idx1-ubyte");

   

   // let mut network = neuralNetwork::NeuralNetwork::new(vec![784,100,10]);
    

    let train_data:Vec<(Array2<f64>, Array2<f64>)> = zipData(train_image, train_label);
    let test_data:Vec<(Array2<f64>, Array2<f64>)> = zipData(test_image, test_label);
    for (x, y) in train_data {
         println!("{}",dnn::neuralNetwork::max(&y));
    }
  }

  fn zipData(train_data:Array2<f64>, train_label:Array2<f64>) -> Vec<(Array2<f64>,Array2<f64>)> {
    let cols = train_data.ncols();
    let rows = train_data.nrows();
    if train_data.nrows() != train_label.nrows() {
        panic!("train_data columns {} not equals train_label columns {} ", train_data.ncols(), train_label.ncols());
    }
    let mut td:Vec<(Array2<f64>,Array2<f64>)> = Vec::new();
    //println!("{:?} ", train_data.row(0));
    for i in 0..rows {
        let mut train_data_row = Array2::<f64>::zeros((cols,1));
        let mut train_label_row = Array2::<f64>::zeros((10,1));

        train_data_row.column_mut(0).assign(&(train_data.row(i).t()));
        train_label_row.column_mut(0).assign(&(train_label.row(i).t()));
        //train_data_row.t();
        //train_label_row.t();
        td.push((train_data_row, train_label_row));
    }
    td
}
  
}