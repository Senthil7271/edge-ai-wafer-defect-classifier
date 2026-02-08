import tf2onnx

SAVED_MODEL_DIR = "saved_model"
ONNX_PATH = "final_model.onnx"

print("Converting SavedModel to ONNX...")

tf2onnx.convert.from_saved_model(
    SAVED_MODEL_DIR,
    output_path=ONNX_PATH,
    opset=13
)

print("Saved:", ONNX_PATH)
