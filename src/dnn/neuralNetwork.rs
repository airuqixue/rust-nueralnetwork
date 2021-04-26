use rand::prelude::*;
use ndarray_rand::rand_distr::{StandardNormal, Normal};
use ndarray_rand::RandomExt;
use ndarray::{Array,Array2,ArrayBase};
use rand::Rng;
#[warn(non_snake_case)]

pub fn shuffle(slice: &mut Vec<(Array2<f64>, Array2<f64>)>) {
    let mut rng = rand::thread_rng();

    let len = slice.len();
    for i in 0..len {
        let next = rng.gen_range(i..len);
        // let (tmpx,tempy) = slice[i];
        // slice[i] = slice[next];
        // slice[next] = (tmpx,tempy);
        slice.swap(i,next);
    }
}

pub struct NeuralNetwork {
	sizes: Vec<usize>,
	weights: Vec<Array2<f64>>,
	biases: Vec<Array2<f64>>,
}

impl<'a> NeuralNetwork{
	pub fn new(sizes:Vec<usize>) -> Self{
	  let mut biases:Vec<Array2<f64>> = Vec::new();
	  let mut weights:Vec<Array2<f64>> = Vec::new();
 		for i in 1..sizes.len() {
 			biases.push(Array::random((sizes[i],1), StandardNormal));
 			let w:Array2<f64> = Array::random((sizes[i], sizes[i-1]), Normal::new(0.0, 1.0/(sizes[i-1] as f64).sqrt()).unwrap());
 			weights.push(w);

 		}
	  NeuralNetwork{
	  	    sizes: sizes,
	  		weights: weights,
	  		biases: biases,
	  }
	}
	// pub fn linear(a: &Array2<f64>, w: &Array2<f64>) -> Array2<f64> {
	// 	w.dot(a)
	// }
	
	pub fn forward(&self, a: &Array2<f64>) -> Array2<f64> {
		//let mut input = Array2::<f64>::zeros((a.nrows(),a.ncols()));
		//input.assign(a);
		let mut input = a.clone();
		// for (w, b) in zip(&self.weights, &self.biases) {
		for i in 0..(self.sizes.len()-1){
			input = sigmoid(&(self.weights[i].dot(&input) + &self.biases[i]));
			// println!("forward a is {:?}", input);
			
		}
		input
	}

	

	pub fn SGD(&mut self, 
		mut training_data:Vec<(Array2<f64>, Array2<f64>)>, 
	    epochs:u32, 
		mini_batch_size:usize, 
		eta:f64, 
		lbda:f64,
		test_data:Vec<(Array2<f64>, Array2<f64>)> ) {
		//println!("w:{:?}", self.weights);
		println!("epochs:{}, batch_size:{},eta:{}, lambda:{}, test_len:{}", epochs,mini_batch_size,eta, lbda, test_data.len());
		//println!("weights_data:{:?}", self.weights[1]);
		let mut maxRatio:usize = 0;
		let mut n_test = 0;
		if test_data.len() > 0 {
			n_test =test_data.len();
		}
		let n_data = training_data.len();
		for i in 0..epochs {
			//shuffle
			//println!("tb: {:?}", training_data[1].1.t());
			shuffle(&mut training_data);
			//println!("ta: {:?}", training_data[1].1.t());
			//println!("after shuffle");
			//let mut sum:f64 =0.0;
			for k in (0..n_data).filter(|x|(x%mini_batch_size == 0)) {
				let mut end = k+mini_batch_size;
				if end > n_data {
					end = n_data;
				} 
			  
			  let mini_batch: &[(Array2<f64>, Array2<f64>)] = &training_data[k..end];
			  //println!("mini_batch range {},{}", k, end);
			  self.update_mini_batch(mini_batch, eta, lbda, n_data);
			 // println!("weights_data:{:?}", self.weights[1]);
			  // for (x, y) in mini_batch {
			  // 	sum += cost(&self.forward(&x), y);
		   //    }
		   //    println!("mini_batch size {}", mini_batch.len());
		   //    println!("Cost is {}",sum);
			  
		    }
		   
		   
		    if n_test > 0 {
		    	let maxr = self.evaluate(&test_data);
		    	if(maxr > maxRatio){
		    		maxRatio = maxr;
		    	}
		    	println!("Epoch {}:{}/{}-max:{}", i, maxr, n_test,maxRatio);
		    }else{
		    	println!("Epoch {} complete", i);
		    }
		}
	}

	pub fn update_mini_batch(&mut self, mini_batch: & [(Array2<f64>, Array2<f64>)], eta:f64, lbda:f64, n:usize) {
		let mut nabla_w:Vec<Array2<f64>> = Vec::new();
		let mut nabla_b:Vec<Array2<f64>> = Vec::new();
		
		for i in 0..(self.sizes.len()-1) {
			nabla_w.push(Array2::<f64>::zeros((self.weights)[i].raw_dim()));
			nabla_b.push(Array2::<f64>::zeros((self.biases)[i].raw_dim()));
		}

		for (x,y) in mini_batch {
            let   (delta_nabla_b, delta_nabla_w) = self.backprop(x,y);
            
            for j in 0..(self.sizes.len()-1) {
            	nabla_w[j] +=   &delta_nabla_w[j];
            	nabla_b[j] +=   &delta_nabla_b[j];  	
            }
           
            
           // nabla_b += delta_nabla_b;
		}
		//println!("finished all backprop");
		//w' = w - (eta/n)nw
		for i in 0..(self.sizes.len()-1) {
			let w = (&((1.0-eta*(lbda/(n as f64)))* &self.weights[i]) - &((eta/(mini_batch.len() as f64)) * &nabla_w[i]));
			let b = (&self.biases[i] - &((eta/(mini_batch.len() as f64)) * &nabla_b[i]));
			//self.weights[i].assign(& w );
			//self.biases[i].assign(& b);
			self.weights[i] = w;
			self.biases[i] = b;
			
		}

	}

	pub fn evaluate(&self, test_data: &Vec<(Array2<f64>,Array2<f64>)>)->usize{
		let mut c:usize = 0;
		//let mut idx:usize = 0;
		for (x,y) in test_data {
			//idx += 1;
			let m = max(&self.forward(&x));
			//println!("{}-{}",idx,m);
			if m == max(&y) {
				c += 1;
			}
		}
		c
	}

	pub fn backprop(&self, x: &Array2<f64>, y: &Array2<f64>) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
		let mut nabla_w:Vec<Array2<f64>> = Vec::new();
		let mut nabla_b:Vec<Array2<f64>> = Vec::new();
		for i in 0..(self.sizes.len()-1) {
			nabla_w.push(Array2::<f64>::zeros((self.weights)[i].raw_dim()));
			nabla_b.push(Array2::<f64>::zeros((self.biases)[i].raw_dim()));
		}
		//feedforward
		let mut activation = x.clone();//Array2::<f64>::zeros((x.nrows(), x.ncols()));
		//activation.assign(x);
		let mut activations = vec![activation];
		let mut zs:Vec<Array2<f64>> = Vec::new();


		 for (b, w) in zip(&self.biases, &self.weights){
			let z = w.dot(&activations[activations.len()-1]) + b;
			zs.push(z);
			activation = sigmoid(&zs[zs.len()-1]);
			activations.push(activation);
		}
		//backward pass
		let mut delta:Array2<f64> = (&activations[activations.len()-1] - y);
       // delta.assign(&);
        let size = nabla_b.len();
		nabla_b[size-1] = delta.clone();

		//let mut delta_w:Array2<f64> = Array2::<f64>::zeros((delta.nrows(), delta.ncols()));
		//delta_w.assign(&delta);
		let size = nabla_w.len();
		//println!("activations lenis {}", activations.len());
		nabla_w[size-1] = (&nabla_b[size-1]).dot(&activations[activations.len()-2].t());

		for l in (0..(self.sizes.len()-2)).rev() {
			//let mut z = Array2::<f64>::zeros((zs[l].nrows(), zs[l].ncols()));
			//z.assign(&zs[l]);
			let sp = sigmoid_prime(&zs[l]);
			delta = self.weights[l+1].t().dot(&delta)*&sp;
			// nabla_b[l].assign(&delta);
			nabla_b[l] = delta.clone();
			//let mut delta_copy = Array2::<f64>::zeros((delta.nrows(), delta.ncols()));
			//delta_copy.assign(&delta);
			nabla_w[l] = (&nabla_b[l]).dot(& activations[l].t());
		}
		(nabla_b, nabla_w)
	}

}


pub fn max(list: & Array2<f64>) -> usize {
	let mut max:f64 = 0.0;
	let mut idx = 0;
	for (p,e) in list.iter().enumerate() {
		if *e > max {
			max = *e;
			idx = p;
		}
	}
	idx
}

pub fn  zip<'a>(a: &'a Vec<Array2<f64>>, b:&'a Vec<Array2<f64>>) -> Vec<(&'a Array2<f64>, &'a Array2<f64>)> {
		let mut tup: Vec<(&Array2<f64>, &Array2<f64>)> = Vec::new();
		let len ={
		 if a.len() > b.len(){
		 	b.len()
		 }else {
		 	a.len()
		 }
		};
		for i in 0..len {
			tup.push((&a[i],&b[i]));
		}
		tup
	}
pub fn sigmoid(z: &Array2<f64>) -> Array2<f64> {
		1.0/(1.0+ z.map(|a| (-a).exp()))
	}

fn sigmoid_prime(z:&Array2<f64>) -> Array2<f64> {
	sigmoid(z) * (1.0 - sigmoid(z))
}

fn cost(a:& Array2<f64>, y:& Array2<f64>) -> f64{
	//println!("a is {:?}", a);
	let mut b=(-y * a.mapv(|e| e.ln()) - (1.0-y)* ((1.0-a).map(|x| x.ln())));
	//println!("cost map: {:?}", b);
	b.iter().filter(|n|!n.is_nan()&& !n.is_infinite()).sum()

}