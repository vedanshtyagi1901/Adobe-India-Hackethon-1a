# 📄 PDF to JSON Outline Processor

This project contains a Python script `process_pdfs.py` that reads all PDF files from the `/app/input` directory, generates a JSON outline (currently using dummy data), and saves the corresponding output in the `/app/output` directory.
---

## 🔄 Input → Output Mapping
/app/input/FileName.pdf ─────────► /app/output/FileName.json

## 🛠️ Build the Docker Image

Make sure you're in the project root (where the `Dockerfile` exists), then run:

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

## 🛠️ Run the Docker Image

Make sure you're in the project root (where the `Dockerfile` exists), then run:

```bash
docker run --rm -v "${PWD}\input:/app/input" -v "${PWD}\output:/app/output" --network none mysolutionname:somerandomidentifier
```