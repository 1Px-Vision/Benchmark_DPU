import os, time
import numpy as np
from pynq_dpu import DpuOverlay

# ───────── Config (edit paths only) ─────────
os.environ.setdefault("XLNX_DPU_TIMEOUT", "60000") # 60s to avoid spurious timeouts
BIT_PATH = "/home/ubuntu/pynq_jupyter_notebooks/pynq-dpu/dpu.bit"
XMODEL_PATH= "/home/ubuntu/pynq_jupyter_notebooks/pynq-dpu/rf_model_2.xmodel"
RF_INPUT_PATH = "/home/ubuntu/pynq_jupyter_notebooks/pynq-dpu/rf_input.npy"
NUM_FRAMES = 1 # set None for all

# ───────── Helpers ─────────
def infer_layout(dims):
"""Return 'NHWC' or 'NCHW' (heuristic). dims is runner input dims."""
# Typical dims are 4D: (B, H, W, C) or (B, C, H, W)
if len(dims) != 4:
return "UNKNOWN"
b, d1, d2, d3 = dims
# If last dim is small (#channels), assume NHWC
if d3 in (1, 2, 3, 4, 8):
return "NHWC"
# If second dim is small, assume NCHW
if d1 in (1, 2, 3, 4, 8):
return "NCHW"
# fallback: prefer NCHW (most common on DPU)
return "NCHW"

def to_layout(batch_nhwc, target, in_dims):
"""
batch_nhwc: (B,H,W,C) ndarray
target: 'NHWC' or 'NCHW'
in_dims: runner dims; used to sanity-shape
"""
if target == "NHWC":
out = batch_nhwc
elif target == "NCHW":
out = np.transpose(batch_nhwc, (0, 3, 1, 2))
else:
out = batch_nhwc
# Force exact dims for the compiled graph (except batch which we pad)
expect = (in_dims[1], in_dims[2], in_dims[3]) if target == "NCHW" else (in_dims[1], in_dims[2], in_dims[3])
# Only check rank compatibility; batch will be handled by pad
if out.ndim != 4:
raise ValueError(f"Expected 4D tensor after layout, got {out.shape}")
return np.ascontiguousarray(out, dtype=out.dtype)

def quantize_int8(x_float, scale):
# DPU int8 convention: value_int8 = round(value_float / scale), clip to [-128,127]
q = np.round(x_float / scale).astype(np.int32)
np.clip(q, -128, 127, out=q)
return q.astype(np.int8, copy=False)

def dequantize_int8(x_int8, scale):
return x_int8.astype(np.float32) * scale

def softmax_stable(logits):
logits = logits.astype(np.float32, copy=False)
m = np.max(logits, axis=1, keepdims=True)
np.subtract(logits, m, out=logits)
np.exp(logits, out=logits)
s = np.sum(logits, axis=1, keepdims=True)
return logits / (s + 1e-12)

# ───────── Load overlay & runner ─────────
overlay = DpuOverlay(BIT_PATH)
overlay.load_model(XMODEL_PATH)
runner = overlay.runner
print("[DPU] Runner")
in_t = runner.get_input_tensors()[0]
out_t = runner.get_output_tensors()[0]

in_dims = tuple(in_t.dims) # e.g. (B,C,H,W) or (B,H,W,C)
out_dims = tuple(out_t.dims) # e.g. (B, num_classes)
in_dtype = str(in_t.dtype).lower() # 'int8' or 'float32'
out_dtype = str(out_t.dtype).lower()
in_scale = getattr(in_t, "scale", 1.0) or 1.0
out_scale = getattr(out_t, "scale", 1.0) or 1.0

compiled_bs = int(in_dims[0]) if in_dims[0] > 0 else 1
layout = infer_layout(in_dims)

print(f"[DPU] in_dims={in_dims} out_dims={out_dims} layout={layout}")
print(f"[DPU] in_dtype={in_dtype} in_scale={in_scale} out_dtype={out_dtype} out_scale={out_scale}")
print(f"[DPU] compiled batch={compiled_bs}")

# ───────── Load data ─────────
X = np.load(RF_INPUT_PATH) # expected shape like (N, 1024, 1, 2) as float32
if NUM_FRAMES is not None:
X = X[:NUM_FRAMES]

# Normalize to float32 NHWC
if X.ndim != 4:
raise ValueError(f"Expected 4D input array, got {X.shape}")
X = X.astype(np.float32, copy=False)

# If the stored shape looks like (N, H, W, C) already, keep it; else try to coerce.
# (User’s dataset was (N,1024,1,2) which is NHWC-ish.)
B, H, W, C = X.shape
X_nhwc = X # assume NHWC
X_nhwc = np.ascontiguousarray(X_nhwc, dtype=np.float32)

# ───────── Inference (fixed batch, padded) ─────────
num_samples = X_nhwc.shape[0]
num_classes = int(np.prod(out_dims[1:])) if len(out_dims) > 1 else 1
pred_logits = np.empty((num_samples, num_classes), dtype=np.float32)
print(f"[DPU] Samples={num_samples}")
i = 0
t0 = time.time()
while i < num_samples:
# Always allocate full compiled batch
in_buf = [None]
out_buf = [None]

# Prepare batch slice (pad with zeros if last chunk)
take = min(compiled_bs, num_samples - i)
batch_nhwc = np.zeros((compiled_bs, H, W, C), dtype=np.float32)
batch_nhwc[:take] = X_nhwc[i:i+take]

# Layout conversion
batch_for_dpu = to_layout(batch_nhwc, layout, in_dims)

# Quantize if needed
if 'int8' in in_dtype:
batch_for_dpu = quantize_int8(batch_for_dpu, in_scale)
else:
batch_for_dpu = batch_for_dpu.astype(np.float32, copy=False)

# Allocate exact shapes the runner expects (compiled_bs fixed)
# Note: we match in/out dims except batch which is compiled_bs
in_shape_exact = (compiled_bs,) + tuple(in_dims[1:])
out_shape_exact = (compiled_bs,) + tuple(out_dims[1:])

in_buf[0] = np.empty(in_shape_exact, dtype=np.int8 if 'int8' in in_dtype else np.float32)
out_buf[0] = np.empty(out_shape_exact, dtype=np.int8 if 'int8' in out_dtype else np.float32)

# Fill input
in_buf[0][...] = 0
in_buf[0][:take, ...] = np.ascontiguousarray(batch_for_dpu, dtype=in_buf[0].dtype)

print(len(in_buf),len(out_buf))

# Execute
print(f"[DPU] Pre-Execute")
jid = runner.execute_async(in_buf, out_buf)
print(f"[DPU] Post-Execute")
try:
runner.wait(jid)
except Exception as e:
# Provide rich context if a CU timeout or similar occurs
raise RuntimeError(
f"DPU execution failed: {e}\n"
f"in_buf shape={in_buf[0].shape} dtype={in_buf[0].dtype} "
f"out_buf shape={out_buf[0].shape} dtype={out_buf[0].dtype} "
f"layout={layout} compiled_bs={compiled_bs}"
)

# Extract logits (dequantize if needed)
raw = out_buf[0][:take]
if 'int8' in out_dtype:
raw = dequantize_int8(raw, out_scale)
logits = raw.reshape(take, -1).astype(np.float32, copy=False)
pred_logits[i:i+take] = logits
i += take

dt = time.time() - t0
print(f"[DPU] Inference OK: {num_samples} samples in {dt:.3f}s")

# ───────── Post-process ─────────
probs = softmax_stable(pred_logits)
pred = np.argmax(probs, axis=1)
print("Pred indices:", pred[: min(10, len(pred))])

# ───────── Cleanup ─────────
del runner
del overlay

