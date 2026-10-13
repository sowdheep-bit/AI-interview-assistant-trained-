# AI Interview Assistant — Adaptive LLM Interview System

An adaptive technical interview assistant powered by a LoRA fine-tuned **Qwen/Qwen1.5-1.8B** model.  
It asks questions from a curated dataset, scores your answers using the local LLM, and dynamically adjusts difficulty (Easy → Medium → Hard) based on your performance.

---

## ⚠️ Before You Start

After cloning, the following items are **not included** in the repo (excluded via `.gitignore`):

| Missing Item | Why It's Missing | What To Do |
|---|---|---|
| `venv/` folder | Machine-specific, too large | Create it yourself (Step 2 below) |
| `.env` file | Contains secret API keys | Create it yourself (Step 4 below) |

The **base model** (`Qwen/Qwen1.5-1.8B`, ~4 GB) will be **downloaded automatically** from HuggingFace the first time you run `app.py`.

The **LoRA adapter weights** (`models/fine_tuned_interviewer/`) are included in this repository and require no additional download.

---

## Project Structure

```
AI-interview-assistant-trained-/
├── data/
│   └── LLM_Interview_Dataset-v2.csv   # Question bank
├── models/
│   └── fine_tuned_interviewer/        # LoRA adapter weights (included)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── chat_template.jinja
│       ├── tokenizer.json
│       └── tokenizer_config.json
├── notebooks/
│   └── training_notebook.ipynb        # Colab fine-tuning notebook
├── scripts/
│   └── evaluate_model.py              # Evaluation script (requires Gemini API key)
├── app.py                             # Main CLI application
├── requirements.txt                   # Python dependencies
├── setup.bat                          # One-click Windows setup script
└── .gitignore
```

---

## Step-by-Step Setup (Windows)

### Step 1 – Prerequisites

Make sure you have the following installed:

- **Python 3.9 or higher** – [Download here](https://www.python.org/downloads/)  
  ✅ During installation, check **"Add Python to PATH"**
- **Git** – [Download here](https://git-scm.com/downloads)

Verify your Python version:
```powershell
python --version
# Expected output: Python 3.9.x or higher
```

---

### Step 2 – Set Up the Python Virtual Environment

> **Option A – Automated (Recommended for Windows)**
```bat
setup.bat
```
Double-click `setup.bat` in File Explorer, or run it in the terminal. It will create the `venv/` folder and install all dependencies automatically.

> **Option B – Manual**
```powershell
# Navigate to the project folder first
cd path\to\AI-interview-assistant-trained-

# Create the virtual environment
python -m venv venv

# Activate it
venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

You should see `(venv)` at the start of your terminal prompt after activation.

> ⚠️ **If you see an error about script execution policy**, run this first:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### Step 3 – LoRA Adapter Weights

The fine-tuned adapter weights are **included in this repository** under `models/fine_tuned_interviewer/`. No additional steps are needed — simply clone the repo and they will be available.

> ℹ️ The **base model** (`Qwen/Qwen1.5-1.8B`) is fetched automatically from HuggingFace on first run. This download is approximately **4 GB** — ensure you have sufficient disk space and a stable internet connection.

---

### Step 4 – Configure the `.env` File

> ⚠️ This is only required for the **evaluation scripts** (`scripts/evaluate_model.py`). The main interview app (`app.py`) does **not** need this.

Create a file named `.env` in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

To obtain a free Gemini API key:
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy and paste it into the `.env` file

---

### Step 5 – Run the Interview System

**Activate the virtual environment** (required each time you open a new terminal):
```powershell
venv\Scripts\activate
```

**Run the application:**
```powershell
python app.py
```

**On first run**, HuggingFace will download the base model (`Qwen/Qwen1.5-1.8B`). This takes a few minutes depending on your internet speed. Subsequent runs load the cached model instantly.

---

## How the Interview Works

1. **Pick a Domain** – Choose from available topics (e.g., Machine Learning, Python, Databases)
2. **Easy Round** – Answer 5 easy questions
3. **Medium Round** – Answer 8 medium questions  
   - Score ≥ 70% average → advance to Hard  
   - Score < 70% → interview ends here
4. **Hard Round** – Answer 5 hard questions
5. **Session Report** – View your total score, verdict (STRONG HIRE / LEANING HIRE / NO HIRE), and every question where marks were lost

**Controls:**
- Type your answer and press **Enter** to submit
- Type `quit` and press **Enter** to exit early

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | `venv` not activated or dependencies not installed | Run `venv\Scripts\activate` then `pip install -r requirements.txt` |
| `OSError: Can't load adapter` | `models/fine_tuned_interviewer/` folder is missing or corrupted | Re-clone the repository |
| `FileNotFoundError: data/LLM_Interview_Dataset-v2.csv` | Working directory is incorrect | Run `python app.py` from the project root folder |
| HuggingFace download stalls | Slow or unstable internet connection | Wait and retry — the download resumes from where it stopped |
| `pip install` fails with SSL error | Corporate network or firewall | Try `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt` |
| `venv\Scripts\activate` gives execution policy error | PowerShell security policy | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## System Requirements

| Requirement | Minimum |
|---|---|
| OS | Windows 10/11, macOS, Linux |
| Python | 3.9+ |
| RAM | 8 GB (16 GB recommended) |
| Disk Space | ~5 GB (4 GB base model + adapter + dependencies) |
| GPU | Not required — runs on CPU |
| Internet | Required on first run to download the base model |

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Model inference engine |
| `transformers` | Load Qwen/Qwen1.5-1.8B base model |
| `peft` | Load the LoRA fine-tuned adapter |
| `pandas` | Read the question dataset CSV |
| `accelerate` | Optimized device mapping for CPU inference |
| `sentencepiece` | Tokenizer backend for Qwen |
| `datasets` | HuggingFace datasets utility |
| `huggingface_hub` | Model download from HuggingFace |

---

## Improving Grader Accuracy

The baseline 63.3% figure reflects a 30-case tier-classification benchmark, not general interview-answer accuracy. The application and evaluator load the adapter on the base model recorded in its metadata and use the same `Score:` prompt used during fine-tuning. These fixes should be benchmarked before reporting an improved percentage.

The current grading dataset contains 126 answers from 42 unique questions, concentrated in Data Structures & Algorithms. A balanced, question-level dataset should be generated before retraining; the 30 benchmark questions must not be used as training data.

```powershell
# Requires GEMINI_API_KEY. --resume preserves existing examples.
python scripts/generate_grading_dataset.py --n-per-cell 100 --resume

# Validation is split by question, so reference answers cannot leak between
# train and validation answer variants.
python scripts/train_grading_lora.py --epochs 4 --rank 32

# Benchmark the new adapter on the held-out 30-case suite.
python scripts/evaluate_model.py --adapter-path models/fine_tuned_interviewer_v2
```

To test a new adapter without replacing the current one:

```powershell
$env:INTERVIEW_ADAPTER_PATH = "./models/fine_tuned_interviewer_v2"
python app.py
```

---

## License

This project is intended for academic and research purposes.
