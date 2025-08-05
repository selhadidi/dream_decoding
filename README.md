
from pathlib import Path, PurePosixPath, PurePath

readme_content = """# Dream-EEG-Decoding 🧠🌙

A **reproducible, lightweight implementation** of our thesis pipeline for  
decoding high-level dream semantics from scalp EEG.  
The repository focuses on the two supervised research questions submitted to *\<Conference\>*:

* **RQ-1  Topic Classification** – predict one of 13 manual dream topics from a 1-sec EEG epoch  
* **RQ-2  Report-Embedding Reconstruction** – regress the sentence-transformer embedding of a dream report from the same EEG epoch

> **Note** Unsupervised clustering for RQ-3 is omitted here to keep dependencies minimal.

---

## 1 Project layout

