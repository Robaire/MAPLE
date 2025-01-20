# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gdown",
# ]
# ///

import argparse
import os

import gdown

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Model checkpoint download script.",
        description="Automatically downloads the model weights required by FastSAM.",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="File output directory",
        type=str,
        dest="output",
        default="./resources",
    )

    parser.add_argument(
        "--force",
        help="Replace existing files",
        action="store_true",
    )

    args = parser.parse_args()

    # Create a resources folder if it doesn't already exist
    try:
        os.mkdir(os.path.expanduser(args.output))
    except FileExistsError:
        pass

    url = "https://drive.google.com/uc?id=1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv"
    output = os.path.expanduser(args.output) + "/FastSAM-x.pt"

    # If forced, download the file
    if args.force:
        gdown.download(url, output)

    # Check if the weights are already there
    else:
        if not os.path.isfile(output):
            gdown.download(url, output)
        else:
            print(f"A file already exists at {output}.")
            print("Use --force to download anyways.")
