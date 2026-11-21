"""Shared test setup.

Import chromadb (and with it onnxruntime) before anything else. On some Windows machines
onnxruntime crashes with "Windows fatal exception: access violation" if it is loaded after
scikit-learn/scipy, which bring an older copy of the Microsoft C++ runtime DLL. Loading it first
avoids the crash; it changes nothing on Linux/macOS.
"""
try:
    import chromadb  # noqa: F401
except Exception:  # pragma: no cover - tests that need it will report the real error
    pass
