import argparse
import subprocess

from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from PIL import ImageGrab

from texify.output import replace_katex_invalid
import threading
from typing import Optional
from queue import Queue


class ModelLoader:
    def __init__(self):
        self.model = None
        self.processor = None
        self._loading_thread: Optional[threading.Thread] = None
        self._loading_complete = threading.Event()
        self._error_queue = Queue()

    def start_loading(self):
        """Start loading the model in a separate thread."""
        self._loading_thread = threading.Thread(target=self._load_model_and_processor)
        self._loading_thread.start()

    def _load_model_and_processor(self):
        """Internal method to load model and processor."""
        try:
            self.model = load_model()
            self.processor = load_processor()
            self._loading_complete.set()
        except Exception as e:
            self._error_queue.put(e)
            self._loading_complete.set()

    def wait_for_loading(self, timeout: Optional[float] = None) -> bool:
        completed = self._loading_complete.wait(timeout)
        if not completed:
            return False
        # Check if there were any errors during loading
        if not self._error_queue.empty():
            error = self._error_queue.get()
            raise RuntimeError(f"Model loading failed: {str(error)}") from error
        return True

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loading_complete.is_set() and self._error_queue.empty()


def inference_single_image(model_loader, katex_compatible=False):
    # Capture image
    subprocess.run(["xscreenshot", "-m", "selection", "-c", "-s"])
    image = ImageGrab.grabclipboard()

    if image is None:
        raise RuntimeError("Failed to capture image from clipboard")

    # Wait for model to be loaded
    if not model_loader.wait_for_loading():
        raise RuntimeError("Model loading timeout")

    # Perform inference
    text = batch_inference([image], model_loader.model, model_loader.processor)
    if katex_compatible:
        text = [replace_katex_invalid(t) for t in text]
    # Try using xclip (most common method)
    process = subprocess.Popen(
        ["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE
    )
    process.communicate(text[0].encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="OCR an image of a LaTeX equation.")
    parser.add_argument(
        "--katex_compatible",
        action="store_true",
        help="Make output KaTeX compatible.",
        default=False,
    )
    args = parser.parse_args()

    # Create model loader
    loader = ModelLoader()

    # Start loading model in background
    loader.start_loading()

    try:
        # Capture and process image
        # This will wait for model loading if necessary
        inference_single_image(loader, args.katex_compatible)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
