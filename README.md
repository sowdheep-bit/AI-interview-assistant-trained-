# IDP3 – Adaptive LLM Interview System

An adaptive technical interview assistant powered by a LoRA fine-tuned **Qwen/Qwen1.5-1.8B** model.  
It asks questions from a curated dataset, scores your answers using the local LLM, and dynamically adjusts difficulty (Easy → Medium → Hard) based on your performance.

---

## ⚠️ Before You Start – Read This First

After cloning, the following are **NOT included** in the repo (intentionally excluded via `.gitignore`):

| Missing Item | Why It's Missing | What To Do |
|---|---|---|
| `venv/` folder | Too large, machine-specific | Create it yourself (Step 2 below) |
| `models/fine_tuned_interviewer/` | Large binary files | Download separately (Step 3 below) |
| `.env` file | Contains secret API keys | Create it yourself (Step 4 below) |

The **base model** (`Qwen/Qwen1.5-1.8B`, ~4 GB) will be **downloaded automatically** from HuggingFace the first time you run `app.py`.

---

## Project Structure

```
IDP3/
├── data/
│   └── LLM_Interview_Dataset-v2.csv   # Question bank (included in repo)
├── models/
│   └── fine_tuned_interviewer/        # LoRA adapter weights ← MUST BE ADDED (see Step 3)
├── notebooks/
│   └── training_notebook.ipynb        # Colab fine-tuning notebook
├── scripts/
│   └── evaluate_model.py              # Evaluation script (needs Gemini API key)
├── app.py                             # Main CLI application ← RUN THIS
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
- **Git** – [Download here](https://git-scm.com/downloads) *(you likely have this already if you cloned the repo)*

Verify your Python version:
```powershell
python --version
# Should output: Python 3.9.x or higher
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
cd path\to\IDP3

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

### Step 3 – Get the LoRA Model Adapter Weights

The fine-tuned adapter files are **not in the repo**. You have two options:

#### Option A – Get the files from your teammate (Recommended)
Ask **Sooraj** to share the `fine_tuned_interviewer/` folder. It contains these files:
```
models/fine_tuned_interviewer/
├── adapter_config.json
├── adapter_model.safetensors   (~12 MB)
├── chat_template.jinja
├── tokenizer.json
├── tokenizer_config.json
└── README.md
```
Place them exactly at `models/fine_tuned_interviewer/` inside the project root.

#### Option B – Use Google Drive / shared link
If Sooraj has shared a Drive link, download the folder and place it at `models/fine_tuned_interviewer/`.

> ℹ️ The **base model** (`Qwen/Qwen1.5-1.8B`) does **not** need to be downloaded manually – it will be fetched from HuggingFace automatically when you first run `app.py`. This download is ~4 GB, so make sure you have enough disk space and a good internet connection.

---

### Step 4 – Configure the `.env` File

> ⚠️ This is only required for the **evaluation scripts** (`scripts/evaluate_model.py`). The main interview app (`app.py`) does **not** need this.

Create a file named `.env` in the project root:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

To get a free Gemini API key:
1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API Key"**
4. Copy and paste it into the `.env` file

---

### Step 5 – Run the Interview System

**First, activate the virtual environment** (every time you open a new terminal):
```powershell
venv\Scripts\activate
```

**Then run the app:**
```powershell
python app.py
```

**On first run**, HuggingFace will download the base model (`Qwen/Qwen1.5-1.8B`). This takes a few minutes depending on your internet speed. Subsequent runs will load the cached model instantly.

---

## How the Interview Works

1. **Pick a Domain** – Choose from available topics (e.g., Machine Learning, Python, Databases)
2. **Easy Round** – Answer 5 easy questions
3. **Medium Round** – Answer 8 medium questions  
   - Score ≥ 70% average → advance to Hard  
   - Score < 70% → interview ends here
4. **Hard Round** – Answer 5 hard questions
5. **Session Report** – See your total score, verdict (STRONG HIRE / LEANING HIRE / NO HIRE), and every question where marks were lost

**Controls:**
- Type your answer and press **Enter** to submit
- Type `quit` and press **Enter** to exit early

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | venv not activated or dependencies not installed | Run `venv\Scripts\activate` then `pip install -r requirements.txt` |
| `OSError: Can't load adapter` | `models/fine_tuned_interviewer/` folder is missing | Complete Step 3 |
| `FileNotFoundError: data/LLM_Interview_Dataset-v2.csv` | Working directory is wrong | Make sure you're running `python app.py` from the project root folder |
| HuggingFace download stalls | Slow/unstable internet | Wait it out or try again — it resumes from where it stopped |
| `pip install` fails with SSL error | Corporate network / firewall | Try `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt` |
| `venv\Scripts\activate` gives execution policy error | PowerShell security policy | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## System Requirements

| Requirement | Minimum |
|---|---|
| OS | Windows 10/11, macOS, Linux |
| Python | 3.9+ |
| RAM | 8 GB (16 GB recommended) |
| Disk Space | ~5 GB (4 GB base model + 500 MB dependencies) |
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

## Improving grader accuracy

The original 63.3% figure is a 30-case tier-classification benchmark, not
interview-answer accuracy in general. The application and evaluator now load
the adapter on the base model recorded in its metadata and use the same
`Score:` prompt used in fine-tuning. Those fixes must be benchmarked before
claiming an improved percentage.

The checked-in grading set currently contains only 126 answers from 42 unique
questions, concentrated in Data Structures & Algorithms. Generate a balanced,
question-level dataset before retraining; do not use the 30 benchmark questions
as training data.

```powershell
# Requires GEMINI_API_KEY. --resume preserves the existing examples.
python scripts/generate_grading_dataset.py --n-per-cell 100 --resume

# Validation is split by question, so reference answers cannot leak between
# train and validation answer variants.
python scripts/train_grading_lora.py --epochs 4 --rank 32

# Benchmark the new adapter on the held-out 30-case suite.
python scripts/evaluate_model.py --adapter-path models/fine_tuned_interviewer_v2
```

Only promote a new adapter after it reaches the target on the benchmark and a
small, independently human-labelled test set. To try it without replacing the
current adapter:

```powershell
$env:INTERVIEW_ADAPTER_PATH = "./models/fine_tuned_interviewer_v2"
python app.py
```
