# Boids

A [Boids](http://www.red3d.com/cwr/boids/) murmuration playground in Python.

This repo now contains **three** things:

| File | What it is |
| --- | --- |
| `index.html` | **Web version** — the murmuration + camera silhouette running entirely in the browser. Nothing to install; host it and open it. |
| `murmuration_camera.py` | Desktop **camera** GUI (Python + OpenCV) with the same idea. |
| `pynboids.py` | The original matplotlib boids demo (from *[Python Playground](https://nostarch.com/pythonplayground)* by Mahesh Venkitachalam). |

---

## Web version (`index.html`)

A flock of birds murmurates on a white web page. Step in front of your webcam
and the swarm reorganises into your **silhouette**; step away and it melts back
into free flight. Everything — camera, human segmentation
([MediaPipe Selfie Segmentation](https://developers.google.com/mediapipe)),
and the boids — runs **client-side in the browser**. The camera feed never
leaves the page.

### Run it

The browser only grants camera access on a **secure origin** (`https://` or
`localhost`). Pick one:

- **GitHub Pages (easiest):** in the repo, *Settings → Pages → Build and
  deployment → Source: GitHub Actions*. The included workflow publishes the
  site; then open the Pages URL and click **Enable camera & start**.
  (Or *Source: Deploy from a branch* → this branch → `/root`.)
- **Locally:**
  ```bash
  python -m http.server 8000
  # open http://localhost:8000/index.html
  ```

> Opening `index.html` as a `file://` won't work — the camera and the
> segmentation library both require it to be served over the web.

### Controls

On-screen buttons (and keys): **Pause** (`Space`), **Formation on/off** (`f`),
**Preview** the mask (`m`), and a **Birds** slider for the flock size.

---

## Desktop version (`murmuration_camera.py`)

A flock of birds drifts across a white screen in a natural murmuration. Your
laptop camera watches the room:

- **Nobody there** → the birds fly in a free, wandering flock.
- **A person steps in** → the flock is pulled toward the person's outline and
  the swarm of birds **traces the human silhouette**. Step away and it melts
  back into free flight.

### How it works

1. **Boids** — a vectorised Reynolds flock (separation / alignment / cohesion)
   in `numpy`.
2. **Silhouette detection** — the camera frame is turned into a binary
   human mask, using either:
   - **MediaPipe Selfie Segmentation** (crisp, full-body — used automatically
     if `mediapipe` is installed), or
   - **OpenCV background subtraction** gated by the HOG people detector
     (fallback, no extra install — just needs a moment of an empty scene first).
3. **Formation** — points are sampled from the silhouette (favouring its
   outline), each bird is matched to its nearest point, and a steering force
   eases the flock onto the shape. When the person leaves, the force fades and
   the murmuration resumes.

### Install & run

```bash
pip install -r requirements.txt        # numpy, opencv-python (+ optional mediapipe)
python murmuration_camera.py
```

Options:

```bash
python murmuration_camera.py --num-boids 500 --camera 0 --width 960 --height 720
```

### Controls

| Key | Action |
| --- | --- |
| `q` / `Esc` | quit |
| `Space` | pause / resume |
| `c` | (fallback back-end) re-learn the empty background — step out of frame first |
| `m` | cycle silhouette preview: off → mask → camera |
| `[` / `]` | fewer / more birds |
| `f` | freeze / unfreeze the "form the human" pull |
| `h` | toggle the HUD |
| `s` | save a screenshot |

**Tip (background-subtraction mode):** start with an empty scene, press `c`,
then step in. MediaPipe needs no calibration.

---

## Original demo (`pynboids.py`)

Requires `matplotlib`, `numpy`, `scipy`.

```bash
python pynboids.py --num-boids 100
```

Left-click adds a boid; right-click scatters the flock. The default matplotlib
axes are hidden for aesthetics.
