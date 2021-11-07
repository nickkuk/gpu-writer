use criterion::{criterion_group, criterion_main, Criterion};
use gpu_writer::{append_gpu_data, EmptyGpuTable, GpuData, GpuDataIter};
use std::{io::Cursor, time::Duration};

pub fn gpu_writer_bench(criterion: &mut Criterion) {
  let mut group = criterion.benchmark_group("gpu_writer");
  let mut src: Vec<u32> = Vec::with_capacity(1024 * 1024);
  for (i, d) in src.iter_mut().enumerate() {
    *d = i as u32;
  }
  let mut dst = vec![0u8; 16 * 1024 * 1024];
  group.bench_function("slice_write_mut_slice", |b| {
    b.iter(|| {
      let gpu_table = append_gpu_data(EmptyGpuTable, src.as_slice());
      let mut writer = dst.as_mut_slice();
      gpu_table.write_into(&mut writer).unwrap();
    })
  });
  group.bench_function("iter_write_mut_slice", |b| {
    b.iter(|| {
      let gpu_table = append_gpu_data(EmptyGpuTable, GpuDataIter::<u32, _>::from(src.iter().copied()));
      let mut writer = dst.as_mut_slice();
      gpu_table.write_into(&mut writer).unwrap();
    })
  });
  group.bench_function("slice_write_cursor", |b| {
    b.iter(|| {
      let gpu_table = append_gpu_data(EmptyGpuTable, src.as_slice());
      let mut writer = Cursor::new(dst.as_mut_slice());
      gpu_table.write_into(&mut writer).unwrap();
    })
  });
  group.bench_function("iter_write_cursor", |b| {
    b.iter(|| {
      let gpu_table = append_gpu_data(EmptyGpuTable, GpuDataIter::<u32, _>::from(src.iter().copied()));
      let mut writer = Cursor::new(dst.as_mut_slice());
      gpu_table.write_into(&mut writer).unwrap();
    })
  });
  group.finish();
}

criterion_group!(
  name = benches;
  config = Criterion::default()
    .sample_size(64)
    .warm_up_time(Duration::from_secs(5))
    .measurement_time(Duration::from_secs(20));
  targets = gpu_writer_bench
);
criterion_main!(benches);
