use bytemuck::{bytes_of, cast_slice, Pod};
use std::io::{Error, Write};

pub fn append_gpu_data<T: GpuTable, D: GpuData>(gpu_table: T, gpu_data: D) -> impl GpuTable {
  Cons(gpu_data, gpu_table)
}

#[macro_export]
macro_rules! gpu_table {
  ($($data:expr,)+) => { gpu_table!($($data),+) };
  ($($data:expr),*) => {
    {
      let _gpu_table = EmptyGpuTable;
      $(
        let _gpu_table = append_gpu_data(_gpu_table, $data);
      )*
      _gpu_table
    }
  };
}

pub trait GpuData {
  fn size(&self) -> usize;
  fn write_into<W: Write>(self, writer: &mut W) -> Result<(), Error>;
}

pub trait GpuTable: GpuData {
  const DATA_COUNT: usize;
  fn data_size(&self) -> usize;
  fn write_header_into<W: Write>(&self, data_offset: usize, writer: &mut W) -> Result<(), Error>;
  fn write_data_into<W: Write>(self, writer: &mut W) -> Result<(), Error>;
}

pub struct EmptyGpuTable;

impl GpuData for EmptyGpuTable {
  fn size(&self) -> usize {
    0
  }
  fn write_into<W: Write>(self, _writer: &mut W) -> Result<(), Error> {
    Ok(())
  }
}

impl GpuTable for EmptyGpuTable {
  const DATA_COUNT: usize = 0;
  fn data_size(&self) -> usize {
    0
  }
  fn write_header_into<W: Write>(&self, _data_offset: usize, _writer: &mut W) -> Result<(), Error> {
    Ok(())
  }
  fn write_data_into<W: Write>(self, _writer: &mut W) -> Result<(), Error> {
    Ok(())
  }
}

struct Cons<D: GpuData, T: GpuTable>(D, T);

impl<D: GpuData, T: GpuTable> GpuData for Cons<D, T> {
  fn size(&self) -> usize {
    std::mem::size_of::<u32>() * Self::DATA_COUNT + self.data_size()
  }
  fn write_into<W: Write>(self, writer: &mut W) -> Result<(), Error> {
    self.write_header_into(std::mem::size_of::<u32>() * Self::DATA_COUNT, writer)?;
    self.write_data_into(writer)?;
    Ok(())
  }
}

impl<D: GpuData, T: GpuTable> GpuTable for Cons<D, T> {
  const DATA_COUNT: usize = T::DATA_COUNT + 1;
  fn data_size(&self) -> usize {
    self.1.data_size() + self.0.size()
  }
  fn write_header_into<W: Write>(&self, data_offset: usize, writer: &mut W) -> Result<(), Error> {
    self.1.write_header_into(data_offset, writer)?;
    let offset = data_offset + self.1.data_size();
    let offset4 = (offset / std::mem::size_of::<u32>()) as u32;
    writer.write_all(&offset4.to_ne_bytes())?;
    Ok(())
  }
  fn write_data_into<W: Write>(self, writer: &mut W) -> Result<(), Error> {
    self.1.write_data_into(writer)?;
    self.0.write_into(writer)?;
    Ok(())
  }
}

impl<T: Pod> GpuData for &[T] {
  fn size(&self) -> usize {
    std::mem::size_of::<T>() * self.len()
  }
  fn write_into<W: Write>(self, writer: &mut W) -> Result<(), Error> {
    writer.write_all(cast_slice(self))?;
    Ok(())
  }
}

pub struct GpuDataIter<T: Pod, I: Iterator<Item = T>> {
  iter: I,
  size: usize,
}

impl<T: Pod, I: Clone + Iterator<Item = T>> From<I> for GpuDataIter<T, I> {
  fn from(iter: I) -> Self {
    let size = std::mem::size_of::<T>() * iter.clone().count();
    GpuDataIter { iter, size }
  }
}

impl<T: Pod, I: Iterator<Item = T>> GpuData for GpuDataIter<T, I> {
  fn size(&self) -> usize {
    self.size
  }
  fn write_into<W: Write>(self, writer: &mut W) -> Result<(), Error> {
    for data in self.iter {
      writer.write_all(bytes_of(&data))?;
    }
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  fn data_count<T: GpuTable>(_gpu_table: &T) -> usize {
    T::DATA_COUNT
  }
  #[test]
  #[allow(clippy::float_cmp)]
  fn test_gpu_writer() {
    let x: [u32; 3] = [1, 2, 3];
    let y: [f32; 4] = [4.0, 5.0, 6.0, 7.0];
    let z_count: usize = 2;
    let gpu_table = gpu_table![
      GpuDataIter::<u32, _>::from(x.into_iter()),
      &y as &[f32],
      GpuDataIter::<f64, _>::from((0..z_count).map(|_| Default::default())),
    ];
    let data_count = data_count(&gpu_table);
    assert_eq!(data_count, 3);
    assert_eq!(gpu_table.data_size(), 4 * (x.len() + y.len() + 2 * z_count));
    assert_eq!(gpu_table.size(), 4 * (data_count + x.len() + y.len() + 2 * z_count));
    let mut v = vec![0u8; gpu_table.size()];
    let mut writer = std::io::Cursor::new(v.as_mut_slice());
    gpu_table.write_into(&mut writer).unwrap();
    let slice_u32 = cast_slice::<u8, u32>(&v);
    let slice_f32 = cast_slice::<u8, f32>(&v);
    let mut header = [0; 3];
    header[0] = data_count as u32;
    header[1] = header[0] + x.len() as u32;
    header[2] = header[1] + y.len() as u32;
    assert_eq!(slice_u32[0..3], header);
    assert_eq!(slice_u32[3..6], x);
    assert_eq!(slice_f32[6..10], y);
    assert_eq!(slice_u32[10..], [0; 4]);
  }
}
