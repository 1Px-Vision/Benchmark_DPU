from pynq_dpu import DpuOverlay
import numpy as np
import pynq
pynq.Device.active_device.reset()

BIT = "/home/ubuntu/pynq_jupyter_notebooks/pynq-dpu/dpu.bit"
XMD = "/home/ubuntu/pynq_jupyter_notebooks/pynq-dpu/rf_model_2.xmodel"

ol = DpuOverlay(BIT)
ol.load_model(XMD)
dpu = ol.runner
print(f"[DPU] Runner")
in_t = dpu.get_input_tensors()[0]
out_t = dpu.get_output_tensors()[0]

din  = np.int8  if 'int8'  in str(in_t.dtype).lower()  else np.float32
dout = np.int8  if 'int8'  in str(out_t.dtype).lower() else np.float32
bs   = in_t.dims[0] if in_t.dims[0] > 0 else 1

dummy = np.zeros((bs, *in_t.dims[1:]), dtype=din)
in_buf  = [dummy]
out_buf = [np.empty((bs, *out_t.dims[1:]), dtype=dout)]

print(f"[DPU] Pre-execute")
jid = dpu.execute_async(in_buf, out_buf)
print(f"[DPU] Post-execute")
dpu.wait(jid)
print("Dummy inference ok. OUT shape:", out_buf[0].shape, "dtype:", out_buf[0].dtype)
