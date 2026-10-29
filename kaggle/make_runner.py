# Kaggle n'expose pas le notebook exécuté par son API : on envoie à la place un script qui
# embarque le notebook, l'exécute avec nbconvert et l'écrit dans /kaggle/working, d'où
# `kaggle kernels output` le récupère avec ses figures.
import base64
from pathlib import Path

HERE = Path(__file__).parent
notebook = (HERE.parent / "notebooks" / "classification_manioc.ipynb").read_bytes()
(HERE / "run_notebook.py").write_text(f'''import base64
import subprocess
from pathlib import Path

path = Path("/kaggle/working/classification_manioc.ipynb")
path.write_bytes(base64.b64decode("{base64.b64encode(notebook).decode()}"))
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                "--ExecutePreprocessor.timeout=-1", str(path)], check=True, cwd="/kaggle/working")
''', encoding="utf-8")
