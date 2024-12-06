import subprocess
import os

def run_command(command, step_name):
    print(f"Current state: {step_name}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error in {step_name}. Exiting...")
        exit(1)

def run_pipeline():

    os.makedirs("results/models", exist_ok=True)

    # run_command("python3 src/create_labels.py", "Creating labels")

    # run_command("python3 src/frame_extraction.py", "Preprocessing videos")

    # run_command("python3 src/signal_extraction.py", "Extracting PPG signals")

    # run_command("python3 src/train.py", "Training the model")

    run_command("python3 src/test.py", "Evaluating the model")

if __name__ == "__main__":
    run_pipeline()
