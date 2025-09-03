# pvl_style_picker.py
import json
from pathlib import Path
from typing import List, Dict

CONFIG_FILENAME = "pvl_style_picker.json"
CONFIG_PATH = Path(__file__).with_name(CONFIG_FILENAME)

_EMBEDDED = [
    {"name": "Ghibli", "prompt": "Studio Ghibli–inspired animation: hand‑painted watercolor/gouache backgrounds, warm pastel palette, soft diffused sunlight, clean thin lines, rounded characters with large expressive eyes, subtle cel‑shading, tranquil whimsical mood, nature‑rich scenes."},
    {"name": "Pixar 3D Animation", "prompt": "Pixar‑quality stylized 3D animation: appealing character proportions, large expressive eyes, soft subsurface scattering, physically based shading with gentle specular highlights, global illumination look, soft area lights with a warm rim light, bright but controlled palette, clean rounded shapes, polished materials, cinematic family‑friendly mood."},
    {"name": "Anime (Shōnen)", "prompt": "High‑energy shōnen anime: crisp lineart, dynamic poses, speed lines, bold cel‑shading, saturated colors, dramatic angles, expressive eyes and brows, action‑packed composition."},
    {"name": "Seinen Manga Ink", "prompt": "Black‑and‑white seinen manga: rich screentones, sharp hatching and cross‑hatching, cinematic panel feel, grounded anatomy, moody shadows, minimal grayscale washes."},
    {"name": "Western Cartoon (Retro)", "prompt": "Mid‑century Western cartoon style: simplified shapes, rubber‑hose limbs, clean thick‑to‑thin outlines, flat colors, limited animation look, cheerful slapstick attitude."},
    {"name": "Ligne Claire (Franco‑Belgian)", "prompt": "Franco‑Belgian ‘ligne claire’: uniform clean outlines, flat bright colors, minimal shading, precise architectural detail, clear readable forms, Tintin‑like clarity."},
    {"name": "American Comic (Silver Age)", "prompt": "Silver Age comic book style: bold inking, halftone dots, primary color palette, dynamic foreshortening, classic superhero staging, punchy onomatopoeia accents."},
    {"name": "Ukiyo‑e Woodblock", "prompt": "Traditional ukiyo‑e woodblock print: flat areas of color, flowing organic linework, stylized waves and clouds, paper grain, muted natural palette, decorative composition."},
    {"name": "Watercolor Storybook", "prompt": "Gentle watercolor picture‑book: soft washes, paper texture, light bleeding edges, pastel tones, loose brushwork, cozy fairytale mood, airy negative space."},
    {"name": "Oil Painting (Impressionist)", "prompt": "Impressionist oil painting: visible brush strokes, impasto texture, broken color, lively edges, warm natural light, plein‑air atmosphere, painterly spontaneity."},
    {"name": "Ink Wash (Sumi‑e)", "prompt": "East Asian ink wash (sumi‑e): expressive calligraphic brushwork, value‑driven forms, soft gradients from wet ink, empty space, poetic minimalism, meditative mood."},
    {"name": "Pastel Chalk Sketch", "prompt": "Soft pastel chalk drawing: velvety texture, smudged edges, tonal layering, paper tooth, gentle color transitions, sketchy line with hand‑drawn charm."},
    {"name": "Pixel Art (16‑bit)", "prompt": "16‑bit pixel art: limited palette, clean 1‑pixel outlines, cluster‑aware shading, subtle dithering, tileable patterns, retro console vibe, crisp sprite readability."},
    {"name": "Isometric Vector", "prompt": "Isometric vector illustration: geometric precision, clean shapes, flat fills with subtle gradients, minimal outlines, consistent 60° axes, tidy iconographic details."},
    {"name": "Cel‑Shaded 3D (Toon)", "prompt": "Cel‑shaded 3D animation look: toon outlines, quantized shading bands, simplified materials, bright stylized palette, clear silhouette, animation‑ready staging."},
    {"name": "Dark Fantasy Painting", "prompt": "Dark fantasy painting: baroque drama, chiaroscuro lighting, ornate textures, gothic motifs, mythic atmosphere, painterly realism without photo references."}
]

_last_mtime = None
_STYLE_LIST: List[Dict[str, str]] = []
_STYLE_MAP: Dict[str, str] = {}

def _use_embedded():
    global _STYLE_LIST, _STYLE_MAP, _last_mtime
    _STYLE_LIST = list(_EMBEDDED)
    _STYLE_MAP = {s["name"]: s["prompt"] for s in _STYLE_LIST}
    _last_mtime = None
    print(f"[PVL_StylePicker] Using embedded styles ({len(_STYLE_LIST)}).")

def _load_styles() -> None:
    global _STYLE_LIST, _STYLE_MAP, _last_mtime
    try:
        if CONFIG_PATH.exists():
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            styles = data.get("styles", [])
            parsed = []
            for item in styles:
                name = str(item.get("name", "")).strip()
                prompt = str(item.get("prompt", "")).strip()
                if name and prompt:
                    parsed.append({"name": name, "prompt": prompt})
            if parsed:
                _STYLE_LIST = parsed
                _STYLE_MAP = {s["name"]: s["prompt"] for s in _STYLE_LIST}
                _last_mtime = CONFIG_PATH.stat().st_mtime
                print(f"[PVL_StylePicker] Loaded {len(_STYLE_LIST)} styles from {CONFIG_PATH.name}.")
                return
    except Exception as e:
        print(f"[PVL_StylePicker] Failed to load JSON: {e}")
    _use_embedded()

def _reload_if_updated() -> None:
    global _last_mtime
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except FileNotFoundError:
        mtime = None
    if mtime != _last_mtime:
        _load_styles()

# Initial load
_load_styles()

class PVL_StylePicker:
    @classmethod
    def INPUT_TYPES(cls):
        _reload_if_updated()
        names = list(_STYLE_MAP.keys())
        if not names:
            _use_embedded()
            names = list(_STYLE_MAP.keys())
        # COMBO dropdown per docs: provide a list[str] as the type
        return {"required": {"style": (names,) }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("style_name", "style_prompt")
    FUNCTION = "pick"
    CATEGORY = "PVL/Style"

    def pick(self, style: str):
        _reload_if_updated()
        prompt = _STYLE_MAP.get(style, "")
        return (style, prompt)

NODE_CLASS_MAPPINGS = {"PVL_StylePicker": PVL_StylePicker}
NODE_DISPLAY_NAME_MAPPINGS = {"PVL_StylePicker": "PVL StylePicker"}
