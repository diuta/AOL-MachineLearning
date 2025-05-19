import pickle
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType

# --- Configuration ---
PICKLE_FILE_PATH = 'musicmood_pipeline.pkl'
ONNX_FILE_PATH = 'pipeline.onnx' # This will be saved in the current directory

def convert_pipeline_to_onnx(pickle_path, onnx_path):
    """Loads a pickled scikit-learn pipeline and converts it to ONNX."""
    print(f"Loading pipeline from {pickle_path}...")
    with open(pickle_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    print("Pipeline loaded.")

    # Define the input type for the ONNX model.
    # The TfidfVectorizer expects a 1D array or list of strings.
    # So, the input will be a tensor of strings, with shape [None, 1] 
    # (batch size None, 1 string feature per instance) or [None] (batch size None, single string)
    # StringTensorType([None, 1]) is common for TF-IDF in skl2onnx.
    # Let's try with [None] first as TF-IDF takes an iterable of strings.
    initial_type = [('string_input', StringTensorType([None]))]
    # If TfidfVectorizer was trained on Series of single strings, this might be [None, 1]
    # For a single string input for prediction, it's usually StringTensorType([1,1]) or StringTensorType([1]) for a batch of one.
    # For variable batch size, [None] or [None,1] is used. Let's use [None] for simplicity, meaning a batch of single strings.

    print("Converting pipeline to ONNX...")
    try:
        onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)
        # target_opset can be adjusted if needed, 12 is a good default.
    except Exception as e:
        print(f"Error during initial conversion attempt: {e}")
        print("Trying with initial_type = [('string_input', StringTensorType([None, 1]))]")
        try:
            initial_type = [('string_input', StringTensorType([None, 1]))]
            onnx_model = convert_sklearn(pipeline, initial_types=initial_type, target_opset=12)
        except Exception as e2:
            print(f"Error during second conversion attempt: {e2}")
            print("Please check the input shape expected by your TfidfVectorizer.")
            print("Common shapes are [None] for a list/Series of strings, or [None, 1] if it expects a 2D array-like of strings.")
            return

    print("Conversion successful.")

    # Save the ONNX model
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"ONNX model saved to {onnx_path}")

if __name__ == '__main__':
    convert_pipeline_to_onnx(PICKLE_FILE_PATH, ONNX_FILE_PATH) 