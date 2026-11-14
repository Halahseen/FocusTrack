"""Download the whole Anti-UAV410 *test* split (all sequences) out of the 9.4 GB zip on
Google Drive, without the train/val parts. zipfile reads the central directory at the end
of the file; each sequence is then fetched with one HTTP Range request.

usage: python fetch_test_split.py OUT_DIR
"""
import io, os, sys, zipfile, time
import requests

URL = "https://drive.usercontent.google.com/download?id=1zsdazmKS3mHaEZWS2BnqbYHPEcIaH5WR&export=download&confirm=t"


class HttpFile(io.RawIOBase):
    def __init__(self, url):
        self.s = requests.Session()
        r = self.s.get(url, headers={"Range": "bytes=0-0"}, allow_redirects=True)
        self.url = r.url
        self.size = int(r.headers["Content-Range"].split("/")[-1])
        self.pos = 0
        self.cache_start, self.cache = 0, b""

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def get(self, start, end):
        for k in range(8):
            try:
                r = self.s.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == end - start + 1:
                    return r.content
            except requests.RequestException as e:
                print("retry", k, e, flush=True)
            time.sleep(2 * (k + 1))
        raise RuntimeError(f"range {start}-{end} failed")

    def prefetch(self, start, end):
        self.cache_start, self.cache = start, self.get(start, end - 1)

    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n == 0 or self.pos >= self.size: return b""
        c0 = self.pos - self.cache_start
        if 0 <= c0 and c0 + n <= len(self.cache):
            self.pos += n
            return self.cache[c0:c0 + n]
        end = min(self.pos + n, self.size) - 1
        data = self.get(self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, b):
        d = self.read(len(b)); b[: len(d)] = d; return len(d)


out = sys.argv[1]
hf = HttpFile(URL)
z = zipfile.ZipFile(hf)
infos = [i for i in z.infolist() if not i.filename.endswith("/")]
seqs = sorted({i.filename.split("/")[1] for i in infos if i.filename.startswith("test/")})
print(len(seqs), "test sequences in the zip", flush=True)
for n, seq in enumerate(seqs, 1):
    sel = [i for i in infos if i.filename.startswith(f"test/{seq}/")]
    dst = os.path.join(out, seq)
    done = os.path.join(dst, ".complete")
    if os.path.exists(done):
        continue
    start = min(i.header_offset for i in sel)
    end = max(i.header_offset + 30 + len(i.filename.encode()) + len(i.extra) + i.compress_size for i in sel) + 4096
    print(f"[{n}/{len(seqs)}] {seq}: {len(sel)} files, {(end - start) / 1e6:.1f} MB", flush=True)
    hf.prefetch(start, min(end, hf.size))
    os.makedirs(dst, exist_ok=True)
    for i in sel:
        with open(os.path.join(dst, os.path.basename(i.filename)), "wb") as f:
            f.write(z.read(i))
    open(done, "w").close()
print("DONE", flush=True)
