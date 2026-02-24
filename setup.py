from setuptools import setup, find_packages

setup(
    name="vrag",
    version="1.0.0",
    description="VRAG: Retrieval-Augmented Video Question Answering for Long-Form Videos",
    author="VRAG Team",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "Pillow>=9.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "opencv-python>=4.7.0",
        "transformers>=4.35.0",
        "open-clip-torch>=2.20.0",
        "faiss-cpu>=1.7.4",
        "rank-bm25>=0.2.2",
        "openai>=1.0.0",
    ],
    extras_require={
        "ocr": ["easyocr>=1.7.0"],
        "whisper": ["openai-whisper>=20231117"],
        "detection": ["ultralytics>=8.0.0"],
        "all": [
            "easyocr>=1.7.0",
            "openai-whisper>=20231117",
            "ultralytics>=8.0.0",
            "scenedetect[opencv]>=0.6.0",
            "sentence-transformers>=2.2.0",
        ],
    },
)
