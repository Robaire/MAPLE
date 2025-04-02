import os
import re
import imageio.v2 as imageio

def sorted_numerically(file_list):
    """Sort filenames by the number embedded in them."""
    return sorted(file_list, key=lambda x: int(re.search(r"frame_(\d+)\.png", x).group(1)))

def make_gif(folder_path, output_filename="output.gif", duration=50):
    """
    Create a GIF from all images in the folder.

    :param folder_path: Path to the folder containing images.
    :param output_filename: Name of the output GIF file.
    :param duration: Duration per frame in milliseconds.
    """
    images = []
    files = [f for f in os.listdir(folder_path) if f.startswith("frame_") and f.endswith(".png")]
    sorted_files = sorted_numerically(files)

    if not sorted_files:
        print("No matching image files found in the folder.")
        return

    for filename in sorted_files:
        img_path = os.path.join(folder_path, filename)
        images.append(imageio.imread(img_path))

    output_path = os.path.join(folder_path, output_filename)
    imageio.mimsave(output_path, images, duration=duration / 1000.0)

    print(f"GIF created successfully: {output_path}")

if __name__ == "__main__":
    folder = input("Enter the folder path: ")
    make_gif(folder)
