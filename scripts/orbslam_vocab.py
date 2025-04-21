#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

import tarfile

if __name__ == "__main__":
    with tarfile.open("resources/ORBvoc.txt.tar.gz", "r:gz") as tar:
        tar.extractall(path="resources")
