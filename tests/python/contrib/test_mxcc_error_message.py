from pathlib import Path

from tvm.contrib import mxcc


def test_error_message_includes_reproducible_command():
    cmd = ["mxcc", "-device-obj", "-o", Path("/tmp/out.mcbin"), "/tmp/in.maca"]
    source = 'extern "C" __global__ void broken() {}'
    message = mxcc._format_mxcc_error(cmd, b"syntax error", source)

    assert "MACA compilation failed." in message
    assert "mxcc -device-obj -o /tmp/out.mcbin /tmp/in.maca" in message
    assert "syntax error" in message
    assert source in message


def test_error_message_replaces_non_utf8_output():
    message = mxcc._format_mxcc_error(["mxcc"], b"\xff", "__kernel")

    assert "Compiler output:" in message
    assert "\ufffd" in message
