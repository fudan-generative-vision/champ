import urllib
import logging
import os
from pathlib import Path
import urllib.request
import tqdm


def download(url: str, output: Path) -> None:
    if not os.path.exists(output.parent):
        os.makedirs(output.parent)

    response = urllib.request.urlopen(url)
    content_length = response.info().get("Content-Length")
    if content_length is None:
        raise ValueError("invalid content length")
    content_length = int(content_length)

    if os.path.exists(output):
        if os.path.getsize(output) == content_length:
            print(f"{output} exists. Download skip.")
            return

    saved_size = 0

    pbar = tqdm.tqdm(total=content_length)
    with open(output, "wb") as f:
        while 1:
            chunk = response.read(8192)
            if not chunk:
                break
            f.write(chunk)
            saved_size += len(chunk)
            pbar.update(len(chunk))

    if saved_size != content_length:
        os.remove(output)
        raise BlockingIOError("fail to download. file cleared")
