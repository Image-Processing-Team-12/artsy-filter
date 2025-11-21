import os
import uuid
import cv2
from flask import Flask, request, jsonify
from pipeline_filters import (
    ComicBookFilter,
    Process,
    CartoonFilter,
    WatercolorFilter,
    CyberpunkFilter,
    ComicBookFilter,
    MangaFilter,
    LineArtFilter
)

# ================================================================
# Flask Setup
# ================================================================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "static/outputs"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ================================================================
# Filter Map
# ================================================================

FILTER_MAP = {
    "cartoon": CartoonFilter,
    "comic": ComicBookFilter,
    "watercolor": WatercolorFilter,
    "cyberpunk": CyberpunkFilter,
    "manga": MangaFilter,
    "lineart": LineArtFilter,
}

# ================================================================
# Routes
# ================================================================

@app.route("/filters", methods=["GET"])
def get_filters():
    return jsonify({
        "available_filters": list(FILTER_MAP.keys())
    })


@app.route("/process", methods=["POST"])
def process_image():

    # Check for uploaded file
    if "file" not in request.files:
        return jsonify({"error": "No file received"}), 400

    img_file = request.files["file"]

    # Filter name
    filter_name = request.form.get("filter", "ghibli")
    block_size = int(request.form.get("pixel_block", 8))

    # Disable pixelization for high-quality styles
    if filter_name in ["ghibli", "watercolor", "cyberpunk"]:
        block_size = 1

    if filter_name not in FILTER_MAP:
        return jsonify({
            "error": f"Unknown filter '{filter_name}'",
            "available_filters": list(FILTER_MAP.keys())
        }), 400

    # Save uploaded image
    filename = f"{uuid.uuid4().hex}.png"
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    img_file.save(input_path)

    print("\n--- RECEIVED FILE ---")
    print("Saved user upload to:", input_path)

    # Create filter instance
    filter_instance = FILTER_MAP[filter_name]()

    # Run processing pipeline
    processor = Process(input_path, filters=[filter_instance], pixel_block=block_size)
    processed_img = processor.run()

    # Save output
    output_filename = f"out_{filename}"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    print("--- SAVING OUTPUT ---")
    print("Saving file to:", output_path)

    success = cv2.imwrite(output_path, processed_img)
    print("Save successful:", success)

    if not success:
        return jsonify({"error": "Could not save output image"}), 500

    # Return JSON
    return jsonify({
        "output_image": f"{OUTPUT_FOLDER}/{output_filename}",
        "filter_used": filter_name,
        "pixel_block": block_size
    })


# ================================================================
# Run Server
# ================================================================

if __name__ == "__main__":
    print("Starting Flask server…")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

