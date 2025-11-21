import cv2
import numpy as np

# ================================================================
#  Artistic Filters
# ================================================================

# --------- 1. Cartoon filter (Anime Style) ----------
class CartoonFilter:
    def apply(self, img):

        # 1) Strong smoothing (makes colors soft & anime-like)
        smooth = img.copy()
        for i in range(8):
            smooth = cv2.bilateralFilter(smooth, 9, 12, 12)

        # 2) Posterization (reduces colors like anime)
        Z = smooth.reshape((-1, 3))
        Z = np.float32(Z)
        K = 10
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, label, center = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        poster = center[label.flatten()].reshape(smooth.shape)

        # 3) Create bold edges
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2
        )
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edges = 255 - edges  # invert → black lines

        # 4) Boost vibrance for anime look
        hsv = cv2.cvtColor(poster, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.multiply(hsv[..., 1], 1.35)
        poster = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 5) Combine edges + colors
        cartoon = cv2.addWeighted(poster, 0.85, edges, 0.15, 0)

        # 6) Final sharpen
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        cartoon = cv2.filter2D(cartoon, -1, kernel)

        return cartoon


# ---------- 2. Watercolor (smooth, brush-stroke style) ----------
class WatercolorFilter:
    def apply(self, img):
        # STEP 1 — Strong watercolor smoothing
        base = cv2.stylization(img, sigma_s=60, sigma_r=0.55)

        # STEP 2 — Edge darkening
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 40, 100)

        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.GaussianBlur(edges, (7,7), 0)

        # Brown watercolor-like edges
        brown = np.zeros_like(img)
        brown[:] = (40, 80, 140)  # BGR brown
        brown_edges = cv2.bitwise_and(brown, brown, mask=edges)

        # STEP 3 — SAFE paper texture (no overflow)
        noise = np.random.normal(0, 6, img.shape).astype(np.float32)
        base_f = base.astype(np.float32)

        paper = base_f + noise
        paper = np.clip(paper, 0, 255).astype(np.uint8)

        # STEP 4 — Final blend
        final = cv2.addWeighted(paper, 0.92, brown_edges, 0.08, 0)

        return final



# ---------- 3. Cyberpunk Neon (saturated, glowing edges) ----------
class CyberpunkFilter:
    def apply(self, img):
        # Step 1: Increase base contrast + saturation
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = cv2.add(hsv[:,:,1], 30)  # saturation boost
        hsv[:,:,2] = cv2.add(hsv[:,:,2], 20)  # brightness boost
        base = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Step 2: Clean thin edge detection
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 60, 140)
        edges = cv2.GaussianBlur(edges, (3,3), 0)  # thin & clean edges

        # Step 3: Convert edges to neon glow (purple-blue)
        neon = cv2.applyColorMap(edges, cv2.COLORMAP_COOL)  # COOL = pink/blue

        # Step 4: Add bloom (glow blur)
        glow = cv2.GaussianBlur(neon, (25,25), 0)

        # Step 5: Blend neon glow with original image
        final = cv2.addWeighted(base, 0.75, glow, 0.25, 0)

        # Step 6: Add light chromatic aberration for cyberpunk vibe
        b,g,r = cv2.split(final)
        r_shift = np.roll(r, 1, axis=1)
        final = cv2.merge([b, g, r_shift])

        return final

    # --------- Comic Book Style Filter ----------
class ComicBookFilter:
    def apply(self, img):

        # 1) Strong color smoothing
        smooth = cv2.bilateralFilter(img, 12, 75, 75)

        # 2) Posterize color (fewer colors = comic style)
        Z = smooth.reshape((-1, 3))
        Z = np.float32(Z)
        K = 8  # fewer clusters than anime
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        ret, label, center = cv2.kmeans(
            Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        poster = center[label.flatten()].reshape(smooth.shape)

        # 3) Strong black outlines
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(gray_blur, 60, 120)
        edges = cv2.dilate(edges, None, iterations=1)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # invert so black lines appear correctly
        edges = 255 - edges  

        # 4) Boost saturation for a comic look
        hsv = cv2.cvtColor(poster, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.4)  # +40% saturation
        comic_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # 5) Combine edges + posterized colors
        comic = cv2.addWeighted(comic_color, 0.90, edges, 0.10, 0)

        return comic

    # --------- Manga Black & White Filter ----------
class MangaFilter:
    def apply(self, img):

        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2) Clean smoothing for nice manga shading
        smooth = cv2.bilateralFilter(gray, 9, 50, 50)

        # 3) Strong outlines like manga ink
        edges = cv2.adaptiveThreshold(
            smooth,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            2
        )

        # invert → black ink on white
        edges_inv = 255 - edges

        # 4) Posterization: reduce grays to manga tones
        # 3 tones = white, gray, black
        levels = [0, 80, 160, 255]
        poster = np.zeros_like(smooth)

        poster[smooth < 70] = 0          # dark ink
        poster[(smooth >= 70) & (smooth < 150)] = 150  # mid-tone
        poster[smooth >= 150] = 255      # white

        # 5) Combine posterized shading + ink outlines
        manga = cv2.bitwise_or(poster, edges_inv)

        # 6) Convert back to 3-channel for saving
        manga_3ch = cv2.cvtColor(manga, cv2.COLOR_GRAY2BGR)

        return manga_3ch

    # --------- Pure Black Line Art Filter ----------
# --------- Line Art PRO (smooth, connected, bold ink) ----------
class LineArtFilter:
    def apply(self, img):

        # 1) Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 2) Apply Gaussian blur to smooth noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3) Difference of Gaussian (DoG) for clean connected edges
        g1 = cv2.GaussianBlur(blur, (0, 0), 1)
        g2 = cv2.GaussianBlur(blur, (0, 0), 2)
        dog = g1 - g2

        # Normalize to 0–255
        dog = cv2.normalize(dog, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

        # 4) Threshold to pure black lines
        _, line = cv2.threshold(dog, 30, 255, cv2.THRESH_BINARY_INV)

        # 5) Morphological operations to CONNECT broken lines
        kernel = np.ones((3, 3), np.uint8)

        # Thicken lines
        line = cv2.dilate(line, kernel, iterations=2)

        # Connect gaps
        line = cv2.morphologyEx(line, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Remove tiny noise
        line = cv2.medianBlur(line, 5)

        # 6) Convert to 3-channel for saving
        lineart = cv2.cvtColor(line, cv2.COLOR_GRAY2BGR)

        return lineart



# ================================================================
#  Processing Pipeline
# ================================================================

class Process:
    def __init__(self, input_path, filters=None, pixel_block=1):
        self.input_path = input_path
        self.filters = filters if filters else []
        self.pixel_block = pixel_block

    def pixelize(self, img):
        if self.pixel_block <= 1:
            return img

        h, w = img.shape[:2]
        x = self.pixel_block
        small = cv2.resize(img, (w // x, h // x), interpolation=cv2.INTER_LINEAR)
        pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        return pix

    def run(self):
        img = cv2.imread(self.input_path)

        # Apply filters first (Ghibli, Watercolor, Cyberpunk)
        for flt in self.filters:
            img = flt.apply(img)

        # Pixelization ONLY if pixel_block > 1
        img = self.pixelize(img)

        return img
