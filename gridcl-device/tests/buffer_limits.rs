//! M2 acceptance: "with an injected limit, an allocation above `max_page_size` returns
//! `AllocationTooLarge`", and "an f64 request on Metal returns `UnsupportedPrecision`".

mod common;

use cubecl::prelude::*;
use gridcl_device::testing::with_properties;
use gridcl_device::{DeviceBuffer, GridError, VECTOR_PAD, check_precision};
use gridcl_testing::{supports, test_client};

/// The injected `max_page_size`: 1024 bytes, 256 u32 elements.
const LIMIT: u64 = 1024;

fn limited(client: &Client) -> cubecl::ir::DeviceProperties {
    let mut props = client.properties().clone();
    props.memory.max_page_size = LIMIT;
    props
}

/// The bytes a buffer of `len` u32 elements allocates: `len` rounded up to `VECTOR_PAD`.
fn padded_bytes(len: usize) -> usize {
    len.max(1).next_multiple_of(VECTOR_PAD) * size_of::<u32>()
}

fn assert_too_large<T>(result: gridcl_device::Result<T>, len: usize, what: &str) {
    match result {
        Err(GridError::AllocationTooLarge { bytes, max }) => {
            assert_eq!(bytes, padded_bytes(len), "{what}: bytes");
            assert_eq!(max, LIMIT, "{what}: max");
        }
        Err(other) => panic!("{what} of {len} elements: {other}, want AllocationTooLarge"),
        Ok(_) => panic!("{what} of {len} elements fits a {LIMIT}-byte limit"),
    }
}

#[test]
fn allocation_above_max_page_size_is_rejected() {
    let client = test_client();
    with_properties(limited(&client), || {
        // 256 elements are exactly 1024 bytes; 241 pad to 256.
        for len in [0, 1, 241, 256] {
            let b = DeviceBuffer::<u32>::uninit(&client, len).expect("uninit within the limit");
            assert_eq!(b.len(), len);
            let data: Vec<u32> = (0..len as u32).collect();
            let b = DeviceBuffer::from_host(&client, &data).expect("from_host within the limit");
            assert_eq!(b.to_host().expect("read back"), data);
        }
        // 257 elements pad to 272: 1088 bytes.
        for len in [257, 1000, 1 << 20] {
            assert_too_large(DeviceBuffer::<u32>::uninit(&client, len), len, "uninit");
            let data = vec![7u32; len];
            assert_too_large(DeviceBuffer::from_host(&client, &data), len, "from_host");
        }
    });
    // Outside the scope the device's own limit applies again.
    DeviceBuffer::<u32>::uninit(&client, 1 << 20).expect("uninit without the injected limit");
}

/// The `zeros` case of the test above. A separate test, because `zeros` runs the fill kernel
/// (T3), while `uninit` and `from_host` are T1.
#[test]
fn allocation_above_max_page_size_is_rejected_zeros() {
    let client = test_client();
    with_properties(limited(&client), || {
        for len in [0, 1, 241, 256] {
            let b = DeviceBuffer::<u32>::zeros(&client, len).expect("zeros within the limit");
            assert_eq!(b.to_host().expect("read back"), vec![0; len]);
        }
        for len in [257, 1000, 1 << 20] {
            assert_too_large(DeviceBuffer::<u32>::zeros(&client, len), len, "zeros");
        }
    });
}

#[test]
fn f64_unsupported_where_device_lacks_it() {
    let client = test_client();
    let supported = supports::<f64>(&client);
    let device = std::env::var("GRIDCL_TEST_DEVICE").unwrap_or_default();
    let result = check_precision::<f64>(&client);
    println!(
        "check_precision::<f64> on {} (GRIDCL_TEST_DEVICE={device:?}): supports_type {supported}, {}",
        client.name(),
        match &result {
            Ok(()) => "Ok".to_string(),
            Err(e) => format!("Err({e})"),
        }
    );
    match device.as_str() {
        "metal" | "metal-native" => assert!(!supported, "Metal must not report f64"),
        "" | "cpu" | "cuda" => assert!(supported, "f64 must be supported on {device:?}"),
        _ => {}
    }
    match result {
        Ok(()) => assert!(supported, "Ok although supports_type(f64) is false"),
        Err(GridError::UnsupportedPrecision { scalar, device }) => {
            assert!(!supported, "UnsupportedPrecision although f64 is supported");
            assert_eq!(scalar, "f64");
            assert_eq!(device, client.name());
        }
        Err(other) => panic!("check_precision::<f64>: {other}"),
    }
    check_precision::<f32>(&client).expect("f32 is supported everywhere");
}
