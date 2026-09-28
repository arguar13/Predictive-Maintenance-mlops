import os

# main.py exige API_KEY al importarse (fail fast). Se fija antes de que
# cualquier test importe el modulo; setdefault respeta un valor externo.
os.environ.setdefault("API_KEY", "test-api-key")
