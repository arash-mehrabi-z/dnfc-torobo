"""
Helper script to find optimal crop coordinates for the neural network.
Run this script and adjust the crop parameters until the region looks good.
"""

import cv2
import numpy as np
from pathlib import Path

# Path to a sample image - adjust this to your actual image path
# You can use an image from your dataset or a screenshot
SAMPLE_IMAGE_PATH = "data/torobo/trajs:360_blocks:3_imgs_2cams/triangle_images_fixed_cam/traj_171/step_000.jpg"

# Initial crop parameters - adjust these values
# Format: (top, left, height, width)
# For 640x480 images with colored blocks in upper-center region
CROP_PARAMS = {
    'top': 210, #180, #80,      # pixels from top edge
    'left': 220, #200, #120,    # pixels from left edge
    'height': 150, #180, #320,  # height of crop region
    'width': 200, #230, #400,   # width of crop region
}


def visualize_crop(image_path, top, left, height, width):
    """Load image and show the crop region."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return None

    img_h, img_w = img.shape[:2]
    print(f"Original image size: {img_w}x{img_h}")

    # Draw rectangle showing crop region on original
    img_with_rect = img.copy()
    cv2.rectangle(img_with_rect,
                  (left, top),
                  (left + width, top + height),
                  (0, 255, 0), 2)  # Green rectangle

    # Add text with coordinates
    text = f"Crop: top={top}, left={left}, h={height}, w={width}"
    cv2.putText(img_with_rect, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    # Add cropped size info
    text2 = f"Cropped size: {width}x{height}"
    cv2.putText(img_with_rect, text2, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Perform the actual crop
    cropped = img[top:top+height, left:left+width]

    # Resize cropped to show at reasonable size
    cropped_display = cv2.resize(cropped, (256, 256))

    return img_with_rect, cropped_display, cropped


def interactive_crop(image_path):
    """Interactive mode to adjust crop with keyboard."""
    params = CROP_PARAMS.copy()
    step = 10  # pixels to move per keypress

    print("\nControls:")
    print("  w/s: move top edge up/down")
    print("  a/d: move left edge left/right")
    print("  i/k: increase/decrease height")
    print("  j/l: increase/decrease width")
    print("  +/-: change step size")
    print("  r: reset to initial values")
    print("  p: print current parameters")
    print("  q: quit and print final parameters")
    print()

    while True:
        result = visualize_crop(image_path, **params)
        if result is None:
            break

        img_with_rect, cropped_display, _ = result

        # Show both windows
        cv2.imshow("Original with crop region", img_with_rect)
        cv2.imshow("Cropped result (resized to 256x256)", cropped_display)

        key = cv2.waitKey(0) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('w'):
            params['top'] = max(0, params['top'] - step)
        elif key == ord('s'):
            params['top'] += step
        elif key == ord('a'):
            params['left'] = max(0, params['left'] - step)
        elif key == ord('d'):
            params['left'] += step
        elif key == ord('i'):
            params['height'] += step
        elif key == ord('k'):
            params['height'] = max(step, params['height'] - step)
        elif key == ord('j'):
            params['width'] = max(step, params['width'] - step)
        elif key == ord('l'):
            params['width'] += step
        elif key == ord('+') or key == ord('='):
            step = min(50, step + 5)
            print(f"Step size: {step}")
        elif key == ord('-'):
            step = max(5, step - 5)
            print(f"Step size: {step}")
        elif key == ord('r'):
            params = CROP_PARAMS.copy()
            print("Reset to initial values")
        elif key == ord('p'):
            print(f"\nCurrent parameters: {params}")
            print(f"Cropped image size: {params['width']}x{params['height']} pixels")

    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("Final crop parameters:")
    print(f"  top = {params['top']}")
    print(f"  left = {params['left']}")
    print(f"  height = {params['height']}")
    print(f"  width = {params['width']}")
    print(f"\nCropped image size: {params['width']}x{params['height']} pixels")
    print("="*50)
    print("\nPyTorch transform code:")
    print(f"""
transforms.Lambda(lambda img: transforms.functional.crop(
    img, top={params['top']}, left={params['left']},
    height={params['height']}, width={params['width']}))
""")

    return params


def batch_test_crop(image_dir, params, num_samples=5):
    """Test crop parameters on multiple images from the dataset."""
    image_dir = Path(image_dir)

    # Find some sample images
    image_files = list(image_dir.glob("**/*.jpg")) + list(image_dir.glob("**/*.png"))

    if not image_files:
        print(f"No images found in {image_dir}")
        return

    # Sample a few images
    import random
    samples = random.sample(image_files, min(num_samples, len(image_files)))

    for img_path in samples:
        result = visualize_crop(str(img_path), **params)
        if result is None:
            continue

        img_with_rect, cropped_display, _ = result

        cv2.imshow(f"Original: {img_path.name}", img_with_rect)
        cv2.imshow("Cropped", cropped_display)

        print(f"Showing: {img_path}")
        print("Press any key for next image, 'q' to quit")

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = SAMPLE_IMAGE_PATH

    print(f"Using image: {image_path}")
    print(f"Initial cropped size: {CROP_PARAMS['width']}x{CROP_PARAMS['height']} pixels")
    interactive_crop(image_path)