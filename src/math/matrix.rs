pub struct Matrix {
	size:(usize,usize),
    data: Box<Vec<Vec<f64>>>,
}

impl Matrix
  {
  	// pub fn new(row:usize, col:usize) -> Self {
  	// 	Matrix{
  	// 		size:(row,col),
  	// 		data:  Box::new(vec![vec![0.0;row];col]),
  	// 	}
  	// }

  	pub fn new(d:Box<Vec<Vec<f64>>>) -> Self {
  		let row = (*(d)).len();
  		let col =  (*d).iter().map(|x|x.len()).max().unwrap();
  		Matrix{
  			size:(row, col),
  			data: d,
  		}
  	}
    pub fn getRow(&self) ->usize {
        (*(self.data)).len()
    }
    pub fn getCol(&self) -> usize {
        (*self.data).iter().map(|x|x.len()).max().unwrap()
    }
    pub fn getUnmutData(& self) -> & Vec<Vec<f64>> {
    	 & (*self.data)
    }
    pub fn getMutData(&mut self) -> &mut Vec<Vec<f64>> {
    	&mut (*self.data)
    }

    pub fn add(&mut self, mut m: Matrix)  {
    	let mut s = self.getMutData();
    	let  mm = m.getUnmutData();
    	let r = m.getRow();
    	let c = m.getCol();
    	for i in 0..r {
    		for j in 0..c {
    			s[i][j] += mm[i][j];
    		}
    	}
    	
    }

}


