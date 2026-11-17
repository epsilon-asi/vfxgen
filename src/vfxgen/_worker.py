"""Frame-render worker process: ``python -m vfxgen._worker`` reads a pickled (payload, times)
from stdin and writes the pickled list of frames to stdout. Running workers as their own
module (instead of multiprocessing's spawn) means calling scripts never need an
``if __name__ == "__main__":`` guard."""
from __future__ import annotations

import pickle
import sys


def main() -> None:
    payload, times = pickle.load(sys.stdin.buffer)
    from .render import _worker_frames, _worker_init
    _worker_init(payload)
    frames = []
    for t in times:                      # one progress tick per frame on stderr
        frames.extend(_worker_frames([t]))
        sys.stderr.buffer.write(b"P\n")
        sys.stderr.buffer.flush()
    pickle.dump(frames, sys.stdout.buffer, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
