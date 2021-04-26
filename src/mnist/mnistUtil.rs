//pub use mod mnistUtil;
//pub use fn readFromFile;

use std::fs::File;
use std::io::prelude::*;
use ndarray::Array2;
use std::mem;
use std::convert::TryInto;

pub fn readFeaturesFromFile(filepath:&str) ->Array2<f64> {
	let mut f  = File::open(filepath).unwrap();
	let mut header_buf = vec![0 as u8;16];

	let header = f.read(&mut header_buf[..]).unwrap();
	
	if header != 16 {
		panic!("can't read header");
	}
	let magic:u32 = read_be_u32(&mut &header_buf[..4]);
	let n = read_be_u32(&mut &header_buf[4..8]);
	let row:u32 = read_be_u32(&mut &header_buf[8..12]);
	let col:u32 = read_be_u32(&mut &header_buf[12..16]);
	println!("n:{}, row:{}, col:{}", n, row, col);

	let size = row*col;
	let mut data:Array2<f64> = Array2::<f64>::zeros((n as usize, size as usize));
	let mut byte_buf = vec![0 as u8; size as usize];

	for i in (0 as usize)..(n as usize) {
		f.read(&mut byte_buf[..]).unwrap();
		for j  in ((0 as usize)..(byte_buf.len() as usize)) {
			
			data[[i,j]] = (byte_buf[j] as f64)/255.0
		}
	}
	data
}

pub fn readLabelsFromFile(filepath:&str) -> Array2<f64> {
	let mut f  = File::open(filepath).unwrap();
	let mut header_buf = vec![0 as u8;8];

	let header = f.read(&mut header_buf[..]).unwrap();
	
	let magic:u32 = read_be_u32(&mut &header_buf[..4]);
	
	let size:u32 = read_be_u32(&mut &header_buf[4..8]);
	
	let mut labels:Array2<f64> = Array2::<f64>::zeros((size as usize, 10 as usize));
	let mut byte_buf = vec![0 as u8;size as usize];
	let n = f.read(&mut byte_buf[..]).unwrap();
	// println!("label size is {}", n);
	for i in (0 as usize)..(size as usize) {
		let label:u32 = byte_buf[i] as u32;
		for j in (0 as usize)..(10 as usize) {
			if (j as u32)==label{
				labels[[i,j]] = 1.0;
			}else{
				labels[[i,j]] = 0.0;
			}
		}
	}
	labels
}

fn read_bb_u32(input: &mut &[u8]) -> u32 {
    let (int_bytes, rest) = input.split_at(std::mem::size_of::<u32>());
    //*input = rest;
    u32::from_be_bytes(int_bytes.try_into().unwrap())
}

fn read_be_u32(input: &mut &[u8]) -> u32 {
	 let (zero_bytes, a) = input.split_at(0);
    //*input = rest;
    u32::from_be_bytes(a.try_into().unwrap())
}