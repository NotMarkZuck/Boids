"""
Murmuration Camera
==================

A live GUI that runs a flock of "boids" (Craig Reynolds' murmuration model)
on a white screen. The laptop camera watches the room:

  * When no person is visible, the birds fly in a free, natural murmuration.
  * When a human figure steps in front of the camera, the flock is pulled
    toward the person's silhouette and reorganises itself so that the swarm
    of birds *traces the shape of the human*.

Two silhouette back-ends are supported, chosen automatically:

  1. MediaPipe Selfie Segmentation  -> crisp, full-body silhouette (best).
  2. OpenCV background subtraction   -> fallback, no extra install.
     (Requires a moment of an empty scene to learn the background.)

Controls (while the window is focused)
--------------------------------------
  q / ESC : quit
  SPACE   : pause / resume the simulation
  c       : (background-subtraction mode) re-learn the empty background
  m       : cycle silhouette preview  (off / mask / camera)
  [ / ]   : fewer / more birds
  f       : toggle formation force on/off (freeze the "form the human" pull)
  h       : toggle the on-screen help / HUD
  s       : save a screenshot (PNG in the current folder)

Run
---
  python murmuration_camera.py
  python murmuration_camera.py --num-boids 500 --camera 0 --width 960 --height 720

Requires: numpy, opencv-python.  Optional (recommended): mediapipe.
"""

import sys
import time
import argparse

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - handled at runtime on the laptop
    sys.exit(
        "OpenCV is required.  Install it with:\n"
        "    pip install opencv-python numpy\n"
        "and (optionally, for the best silhouettes):\n"
        "    pip install mediapipe"
    )


# ---------------------------------------------------------------------------
# Boids flock
# ---------------------------------------------------------------------------
class Flock:
    """A vectorised murmuration.

    All positions/velocities are numpy arrays of shape (N, 2) living in the
    simulation's pixel space (0..width, 0..height).
    """

    def __init__(self, n, width, height):
        self.width = width
        self.height = height
        self._init_agents(n)

        # Flocking radii / strengths -----------------------------------
        self.sep_radius = 18.0     # personal space
        self.align_radius = 45.0   # match heading with these neighbours
        self.cohere_radius = 60.0  # drift toward this neighbourhood's centre

        self.sep_weight = 0.9
        self.align_weight = 0.35
        self.cohere_weight = 0.30

        self.max_speed = 5.0
        self.max_force = 0.4       # steering acceleration cap per rule

        # Silhouette formation -----------------------------------------
        self.targets = None        # (N, 2) target points, or None when free
        self.form_weight = 1.1     # how hard birds are pulled onto the shape
        self.formation_on = True   # user toggle
        # 0 = free murmuration, 1 = fully committed to the silhouette.
        self._blend = 0.0

    # -- setup -----------------------------------------------------------
    def _init_agents(self, n):
        self.n = int(n)
        cx, cy = self.width / 2.0, self.height / 2.0
        self.pos = np.column_stack([
            cx + (np.random.rand(self.n) - 0.5) * self.width * 0.6,
            cy + (np.random.rand(self.n) - 0.5) * self.height * 0.6,
        ]).astype(np.float64)
        angles = np.random.rand(self.n) * 2.0 * np.pi
        self.vel = np.column_stack([np.cos(angles), np.sin(angles)]).astype(np.float64)
        self.vel *= 3.0

    def set_count(self, n):
        n = max(10, int(n))
        if n == self.n:
            return
        self._init_agents(n)
        self.targets = None

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _limit(vectors, max_mag):
        """Clip each row of `vectors` to length `max_mag` (in place-ish)."""
        mag = np.linalg.norm(vectors, axis=1, keepdims=True)
        mag[mag == 0] = 1.0
        scale = np.minimum(1.0, max_mag / mag)
        return vectors * scale

    # -- main step -------------------------------------------------------
    def update(self):
        # Pairwise offset / distance matrices.  Fine for a few hundred boids.
        delta = self.pos[:, None, :] - self.pos[None, :, :]     # (N, N, 2)
        dist2 = np.einsum("ijk,ijk->ij", delta, delta)          # (N, N)
        np.fill_diagonal(dist2, np.inf)

        acc = np.zeros_like(self.pos)

        # RULE 1: separation -- steer away from close crowding.
        sep_mask = dist2 < self.sep_radius ** 2
        with np.errstate(invalid="ignore"):
            weight = sep_mask / np.maximum(dist2, 1e-6)
        sep = np.einsum("ij,ijk->ik", weight, delta)
        acc += self._limit(sep, self.max_force) * self.sep_weight

        # RULE 2: alignment -- match neighbours' heading.
        align_mask = dist2 < self.align_radius ** 2
        counts = align_mask.sum(axis=1, keepdims=True)
        avg_vel = align_mask @ self.vel
        np.divide(avg_vel, counts, out=avg_vel, where=counts > 0)
        align = avg_vel - self.vel
        acc += self._limit(align, self.max_force) * self.align_weight

        # RULE 3: cohesion -- drift toward the local centre of mass.
        coh_mask = dist2 < self.cohere_radius ** 2
        ccounts = coh_mask.sum(axis=1, keepdims=True)
        centre = coh_mask @ self.pos
        np.divide(centre, ccounts, out=centre, where=ccounts > 0)
        cohere = np.where(ccounts > 0, centre - self.pos, 0.0)
        acc += self._limit(cohere, self.max_force) * self.cohere_weight

        # Ease the formation blend toward its goal so shapes assemble and
        # dissolve smoothly instead of snapping.
        want = 1.0 if (self.formation_on and self.targets is not None) else 0.0
        self._blend += (want - self._blend) * 0.08

        if self._blend > 0.01 and self.targets is not None:
            to_target = self.targets - self.pos
            steer = self._limit(to_target, self.max_force * 6.0)
            acc = acc * (1.0 - self._blend) + steer * self.form_weight * self._blend

        # Integrate.
        self.vel += acc
        self.vel = self._limit(self.vel, self.max_speed)
        # Keep a little life even while holding a pose.
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        too_slow = speed < 0.4
        if np.any(too_slow):
            jitter = (np.random.rand(self.n, 2) - 0.5)
            self.vel = np.where(too_slow, self.vel + jitter * 0.5, self.vel)

        self.pos += self.vel
        self._wrap()

    def _wrap(self):
        m = 8.0
        self.pos[:, 0] = np.where(self.pos[:, 0] < -m, self.width + m, self.pos[:, 0])
        self.pos[:, 0] = np.where(self.pos[:, 0] > self.width + m, -m, self.pos[:, 0])
        self.pos[:, 1] = np.where(self.pos[:, 1] < -m, self.height + m, self.pos[:, 1])
        self.pos[:, 1] = np.where(self.pos[:, 1] > self.height + m, -m, self.pos[:, 1])

    # -- targets ---------------------------------------------------------
    def set_silhouette_targets(self, points):
        """Assign every bird a point sampled from the silhouette.

        `points` is an (M, 2) array of pixel coordinates that lie inside the
        detected human shape.  Each bird is matched to the nearest target so
        the flock settles onto the outline without long criss-crossing paths.
        """
        if points is None or len(points) == 0:
            self.targets = None
            return

        # Resample to exactly N points (with replacement if the silhouette
        # gave us fewer than we have birds).
        idx = np.random.choice(len(points), size=self.n,
                               replace=len(points) < self.n)
        chosen = points[idx]

        # Greedy-ish nearest matching: pull each bird toward its closest
        # chosen point.  Cheap and looks organic.
        d = np.linalg.norm(self.pos[:, None, :] - chosen[None, :, :], axis=2)
        nearest = np.argmin(d, axis=1)
        self.targets = chosen[nearest].astype(np.float64)


# ---------------------------------------------------------------------------
# Human silhouette detection
# ---------------------------------------------------------------------------
class SilhouetteDetector:
    """Turns a camera frame into a binary human-silhouette mask.

    Prefers MediaPipe Selfie Segmentation; falls back to OpenCV MOG2
    background subtraction (gated by the HOG people-detector so random
    motion doesn't trigger a "human").
    """

    def __init__(self):
        self.backend = "background-subtraction"
        self._seg = None
        try:
            import mediapipe as mp  # noqa: F401
            self._mp = mp
            self._seg = mp.solutions.selfie_segmentation.SelfieSegmentation(
                model_selection=1
            )
            self.backend = "mediapipe"
        except Exception:
            self._mp = None

        # Fallback machinery.
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False
        )
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self._hog_frame = 0
        self._hog_ok = False

    def relearn_background(self):
        """Reset the background model (fallback back-end only)."""
        self._bg = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=40, detectShadows=False
        )

    # -- public ----------------------------------------------------------
    def detect(self, frame):
        """Return (present, mask) where mask is uint8 0/255 or None."""
        if self.backend == "mediapipe":
            return self._detect_mediapipe(frame)
        return self._detect_bgsub(frame)

    # -- back-ends -------------------------------------------------------
    def _detect_mediapipe(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = self._seg.process(rgb)
        if res.segmentation_mask is None:
            return False, None
        mask = (res.segmentation_mask > 0.5).astype(np.uint8) * 255
        mask = self._clean(mask)
        present = (mask > 0).mean() > 0.02     # at least ~2% of the frame
        return present, (mask if present else None)

    def _detect_bgsub(self, frame):
        fg = self._bg.apply(frame)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        fg = self._clean(fg, strong=True)

        # Confirm it is actually a person every few frames with HOG.
        self._hog_frame += 1
        if self._hog_frame % 5 == 0:
            small = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
            rects, _ = self._hog.detectMultiScale(
                small, winStride=(8, 8), padding=(8, 8), scale=1.05
            )
            self._hog_ok = len(rects) > 0

        coverage = (fg > 0).mean()
        present = self._hog_ok and coverage > 0.02
        return present, (fg if present else None)

    # -- shared ----------------------------------------------------------
    @staticmethod
    def _clean(mask, strong=False):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        if strong:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=3)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    @staticmethod
    def largest_blob(mask):
        """Keep only the biggest connected region (the person)."""
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if num <= 1:
            return mask
        biggest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        return np.where(labels == biggest, 255, 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Sampling target points from a silhouette mask
# ---------------------------------------------------------------------------
def sample_targets(mask, sim_w, sim_h, max_points=1200):
    """Sample points inside the silhouette, mapped into simulation space.

    We favour the *outline* of the shape (so the flock reads as a figure)
    plus a lighter fill of interior points.
    """
    if mask is None:
        return None

    mh, mw = mask.shape[:2]

    # Outline points -- strong emphasis on the shape's edge.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edge_pts = []
    for c in contours:
        if cv2.contourArea(c) < 50:
            continue
        edge_pts.append(c.reshape(-1, 2))
    edge = np.concatenate(edge_pts, axis=0) if edge_pts else np.empty((0, 2))

    # Interior fill points.
    ys, xs = np.where(mask > 0)
    fill = np.column_stack([xs, ys]) if len(xs) else np.empty((0, 2))

    if len(edge) == 0 and len(fill) == 0:
        return None

    def take(arr, k):
        if len(arr) == 0 or k <= 0:
            return np.empty((0, 2))
        idx = np.random.choice(len(arr), size=min(k, len(arr)),
                               replace=len(arr) < k)
        return arr[idx]

    n_edge = int(max_points * 0.6)
    n_fill = max_points - n_edge
    pts = np.concatenate([take(edge, n_edge), take(fill, n_fill)], axis=0)

    # Map camera/mask pixels -> simulation pixels.
    pts = pts.astype(np.float64)
    pts[:, 0] *= sim_w / float(mw)
    pts[:, 1] *= sim_h / float(mh)
    return pts


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def draw_flock(canvas, flock, forming):
    """Draw each bird as a small oriented chevron on the white canvas."""
    body = (60, 60, 60) if not forming else (30, 30, 30)
    speed = np.linalg.norm(flock.vel, axis=1, keepdims=True)
    speed[speed == 0] = 1.0
    heading = flock.vel / speed
    perp = np.column_stack([-heading[:, 1], heading[:, 0]])

    size = 5.0
    tip = flock.pos + heading * size * 1.6
    left = flock.pos - heading * size + perp * size * 0.8
    right = flock.pos - heading * size - perp * size * 0.8

    tri = np.stack([tip, left, right], axis=1).astype(np.int32)
    cv2.fillPoly(canvas, tri, body, lineType=cv2.LINE_AA)


def put_hud(canvas, lines, forming):
    y = 24
    for i, text in enumerate(lines):
        color = (0, 0, 0)
        cv2.putText(canvas, text, (14, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
        y += 22
    badge = "FORMING SILHOUETTE" if forming else "FREE FLIGHT"
    bc = (0, 120, 0) if forming else (120, 120, 120)
    cv2.putText(canvas, badge, (14, canvas.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bc, 2, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Murmuration that forms a human silhouette from the camera.")
    ap.add_argument("--num-boids", type=int, default=400, dest="n")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--width", type=int, default=960, help="window width")
    ap.add_argument("--height", type=int, default=720, help="window height")
    args = ap.parse_args()

    sim_w, sim_h = args.width, args.height

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit(f"Could not open camera {args.camera}. Try a different --camera index.")

    flock = Flock(args.n, sim_w, sim_h)
    detector = SilhouetteDetector()

    win = "Murmuration Camera"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, sim_w, sim_h)

    print(f"Silhouette back-end: {detector.backend}")
    if detector.backend == "background-subtraction":
        print("Tip: step out of frame for a second, press 'c' to learn the empty "
              "background, then step back in.")

    paused = False
    show_help = True
    preview = 0                 # 0 off, 1 mask, 2 camera
    retarget_every = 6          # rebuild targets every N frames
    frame_i = 0
    last = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed; exiting.")
            break
        frame = cv2.flip(frame, 1)   # mirror -> feels natural

        present, mask = detector.detect(frame)
        if present and mask is not None:
            mask = detector.largest_blob(mask)
            if frame_i % retarget_every == 0:
                flock.set_silhouette_targets(
                    sample_targets(mask, sim_w, sim_h, max_points=max(flock.n * 3, 600))
                )
        else:
            flock.targets = None

        if not paused:
            flock.update()

        forming = flock.formation_on and flock.targets is not None

        # --- draw -------------------------------------------------------
        canvas = np.full((sim_h, sim_w, 3), 255, np.uint8)
        draw_flock(canvas, flock, forming)

        if preview and mask is not None:
            thumb_w = sim_w // 4
            thumb_h = sim_h // 4
            if preview == 1:
                pv = cv2.cvtColor(cv2.resize(mask, (thumb_w, thumb_h)),
                                  cv2.COLOR_GRAY2BGR)
            else:
                pv = cv2.resize(frame, (thumb_w, thumb_h))
            canvas[10:10 + thumb_h, sim_w - thumb_w - 10:sim_w - 10] = pv

        # FPS (smoothed).
        now = time.time()
        dt = now - last
        last = now
        if dt > 0:
            fps = fps * 0.9 + (1.0 / dt) * 0.1

        if show_help:
            put_hud(canvas, [
                f"birds: {flock.n}    fps: {fps:4.1f}    backend: {detector.backend}",
                "[q]uit  [space]pause  [c]al-bg  [m]preview  [ ]less/more  [f]orm  [s]hot  [h]ud",
            ], forming)
        else:
            put_hud(canvas, [], forming)

        cv2.imshow(win, canvas)

        # --- input ------------------------------------------------------
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("c"):
            detector.relearn_background()
            print("Background re-learned.")
        elif key == ord("m"):
            preview = (preview + 1) % 3
        elif key == ord("f"):
            flock.formation_on = not flock.formation_on
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("["):
            flock.set_count(flock.n - 50)
        elif key == ord("]"):
            flock.set_count(flock.n + 50)
        elif key == ord("s"):
            fname = f"murmuration_{int(time.time())}.png"
            cv2.imwrite(fname, canvas)
            print("Saved", fname)

        frame_i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
