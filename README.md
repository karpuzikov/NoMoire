# NoMoire

Standalone halftone / moire remover. Dark UI.

```
pip install -r requirements.txt
python descreener_app.py
```

Test: `pip install pytest && pytest`

shortcut: channels are R, G, B (not CMYK inks), upgrade if rosette moire remains.
shortcut: peaks detected on a 1024px center crop, upgrade if the screen varies across the page.
