use neuralnetwork::math::matrix;
use neuralnetwork::dnn::neuralNetwork;
use neuralnetwork::mnist::mnistUtil;
use std::fs::File;
use std::io::prelude::*;
use std::mem;
use std::convert::TryInto;
use ndarray::Array2;
use ndarray::arr2;
//use itertools::Itertools;
fn main() {
	 let train_image = mnistUtil::readFeaturesFromFile("D:\\rust-pj\\neuralnetwork\\src\\mnist\\train-images.idx3-ubyte");
	 let train_label = mnistUtil::readLabelsFromFile("D:\\rust-pj\\neuralnetwork\\src\\mnist\\train-labels.idx1-ubyte");

	 let test_image = mnistUtil::readFeaturesFromFile("D:\\rust-pj\\neuralnetwork\\src\\mnist\\t10k-images.idx3-ubyte");
	 let test_label = mnistUtil::readLabelsFromFile("D:\\rust-pj\\neuralnetwork\\src\\mnist\\t10k-labels.idx1-ubyte");

   

    let mut network = neuralNetwork::NeuralNetwork::new(vec![784,441,10]);
    

    let train_data:Vec<(Array2<f64>, Array2<f64>)> = zipData(train_image, train_label);
   	let  test_data:Vec<(Array2<f64>, Array2<f64>)> = zipData(test_image, test_label);

    network.SGD(train_data, 100, 13, 0.6, 0.73 ,test_data);
    // for (x, y) in train_data {
    //      print!("{}",neuralNetwork::max(&neuralNetwork::forward(&x)));
    // }
    // let mut dt:Vec<(Array2<f64>, Array2<f64>)> = vec![(arr2(&[[2.],[2.]]),arr2(&[[3.],[4.]])),
    // 												(arr2(&[[1.],[1.]]),arr2(&[[2.],[3.]])),
    // 												(arr2(&[[3.],[3.]]),arr2(&[[4.],[5.]]))];
    // neuralNetwork::shuffle(&mut dt);
    // println!("dt0: {:?}", dt[0]);
    // println!("dt1: {:?}", dt[1]);
    // println!("dt2: {:?}", dt[2]);
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

fn shit(){
		for i in (0..10){
		println!("i is {}",i)
	}
   // let mt = matrix::Matrix::new(3,2);
    let data = Box::new(vec![vec![1.0;3];4]);
    let data2 = Box::new(vec![vec![2.0;3];4]);
    
    let mut mt = matrix::Matrix::new(data);
    println!("row {}, col {}", mt.getRow(), mt.getCol());
   
    let mut mt2 = matrix::Matrix::new(data2);

    mt.add(mt2);

    for r in 0..mt.getRow(){
    	for c in 0..mt.getCol(){
    		println!("mt {},{} value is {}",r,c,mt.getUnmutData()[r][c]);
    	}
    }

   let sizes = vec![3,2,1];
   let network = neuralNetwork::NeuralNetwork::new(sizes);

   let n = 100;
   let batch = 13;
   for  mut k in (0..n).filter(|x|(x%batch == 0)) {
   	    
   		println!("k={}",k);
   		k += batch
   }

println!("i32 {} ", std::mem::size_of::<u32>());
  let idata = mnistUtil::readFeaturesFromFile("D:\\rust-pj\\neuralnetwork\\src\\mnist\\train-images.idx3-ubyte");
 // let ilabels = mnistUtil::readLabelsFromFile("D:\\rust-pj\\neuralnetwork\\src\\mnist\\train-labels.idx1-ubyte");
  
}

// fn test() {
// 	let mut f = File::open("D:\\rust-pj\\neuralnetwork\\src\\mnist\\train-images.idx3-ubyte").unwrap();
// 	let mut header_buf = vec![0 as u8;16];
// 	let header = f.read(&mut header_buf[..]).unwrap();
// 	let mut magic_slice = &header_buf[..4];
// 	let magic:u32 = read_be_u32(&mut magic_slice);
	
// 	let row:u32 = read_be_u32(&mut &header_buf[4..8]);
// 	let col:u32 = read_be_u32(&mut &header_buf[8..12]);
// 	let n = read_be_u32(&mut &header_buf[12..16]);
// 	 // println!("magic:{}", magic);
// 	println!("magic:{}, row:{}, col:{}, n:{}", magic, row, col, n);

// }

// fn read_be_u32(input: &mut &[u8]) -> u32 {
//     let (int_bytes, rest) = input.split_at(std::mem::size_of::<u32>());
//     *input = rest;
//     u32::from_be_bytes(int_bytes.try_into().unwrap())
// }
