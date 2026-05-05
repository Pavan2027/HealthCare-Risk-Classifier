import os, sys, traceback

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

log = open("train_log.txt", "w", encoding="utf-8")

try:
    log.write("Step 1: importing os/sys...\n"); log.flush()
    log.write("Step 2: importing training.config...\n"); log.flush()
    import training.config
    log.write("Step 3: importing training.data_loader...\n"); log.flush()
    import training.data_loader
    log.write("Step 4: importing torch...\n"); log.flush()
    import torch
    log.write("Step 5: importing transformers...\n"); log.flush()
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
    log.write("Step 6: importing sklearn...\n"); log.flush()
    from sklearn.metrics import f1_score
    log.write("Step 7: importing training.train...\n"); log.flush()
    from training.train import train
    log.write("Step 8: starting training...\n"); log.flush()

    # Redirect stdout for training
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = log
    sys.stderr = log

    train()

    sys.stdout = old_stdout
    sys.stderr = old_stderr
    log.write("Training complete!\n")
except Exception as e:
    log.write(f"ERROR: {e}\n")
    traceback.print_exc(file=log)
except SystemExit as e:
    log.write(f"SystemExit: {e}\n")
finally:
    log.flush()
    log.close()
