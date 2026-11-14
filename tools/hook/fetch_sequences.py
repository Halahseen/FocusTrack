"""Download selected Anti-UAV410 test sequences out of the 9.4 GB zip on Google Drive
without downloading the whole zip: zipfile reads the central directory at the end of the
file, and every read is served with an HTTP Range request.

usage: python fetch_sequences.py OUT_DIR SEQ [SEQ ...]
"""
import io, os, sys, zipfile
import requests

URL = "https://drive.usercontent.google.com/download?id=1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR&export=download&confirm=t"


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.s = requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True)
        self.url = r.url
        self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.pos = 0

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    cache_start, cache = 0, b""

    def prefetch(self, start, end):
        """Fetch one big contiguous range (a whole sequence) in a single request."""
        r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end - 1}"})
        assert r.status_code == 206, r.status_code
        self.cache_start, self.cache = start, r.content

    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        c0 = self.pos - self.cache_start
        if 0 <= c0 and c0 + n <= len(self.cache):
            self.pos += n
            return self.cache[c0:c0 + n]
        end = min(self.pos + n, self.size) - 1
        for _ in range(5):
            r = self.s.get(self.url, headers={"Range": f"bytes={self.pos}-{end}"})
            if r.status_code == 206: break
        data = r.content
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[: len(d)] = d; return len(d)


out, seqs = sys.argv[1], sys.argv[2:]
hf = HttpFile(URL)
z = zipfile.ZipFile(hf)
infos = z.infolist()
for seq in seqs:
    sel = [i for i in infos if i.filename.startswith(f"test/{seq}/") and not i.filename.endswith("/")]
    members = [i.filename for i in sel]
    start = min(i.header_offset for i in sel)
    end = max(i.header_offset + 30 + len(i.filename.encode()) + len(i.extra) + i.compress_size for i in sel) + 4096
    print(seq, len(members), "files", round((end - start) / 1e6, 1), "MB", flush=True)
    hf.prefetch(start, min(end, hf.size))
    dst = os.path.join(out, seq); os.makedirs(dst, exist_ok=True)
    for m in members:
        p = os.path.join(dst, os.path.basename(m))
        if os.path.exists(p): continue
        with open(p, "wb") as f: f.write(z.read(m))
