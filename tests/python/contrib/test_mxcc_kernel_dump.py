import hashlib

from tvm.contrib import mxcc


def test_kernel_file_name_is_stable_and_content_addressed():
    source = 'extern "C" __global__ void add(float* x) { x[0] += 1.0f; }'
    expected_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]

    assert mxcc._kernel_file_name(source) == f"tvm_kernels_{expected_hash}"
    assert mxcc._kernel_file_name(source) == mxcc._kernel_file_name(source)


def test_kernel_file_name_changes_with_source():
    assert mxcc._kernel_file_name('extern "C" __global__ void a() {}') != mxcc._kernel_file_name(
        'extern "C" __global__ void b() {}'
    )


def test_kernel_file_name_accepts_bytes_and_none():
    source = b'extern "C" __global__ void add(float* x) { x[0] += 1.0f; }'
    expected_hash = hashlib.sha256(source).hexdigest()[:16]

    assert mxcc._kernel_file_name(source) == f"tvm_kernels_{expected_hash}"
    assert mxcc._kernel_file_name(None) == f"tvm_kernels_{hashlib.sha256(b'').hexdigest()[:16]}"
