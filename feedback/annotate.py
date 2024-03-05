import argparse
import pandas as pd
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlabeled_path",
                        default="feedback/unlabeled.csv",
                        help="path the the unlabeled prompt/responseab data")
    parser.add_argument("--labels_path",
                        default="feedback/labels.csv",
                        help="path to save the response data to")
    args = parser.parse_args()


    if not os.path.exists(args.labels_path):
        with open(args.labels_path, "w") as f:
            f.write("label,bad\n")
    
    df = pd.read_csv(args.unlabeled_path)
    n = sum(1 for _ in open(args.labels_path)) - 1
    with open(args.labels_path, "a") as f:
        for i in range(n, len(df)):
            row = df.iloc[i]
            print("".join(["=" for _ in range(100)]))
            print(f"Prompt {i}/{len(df)} ({int(i / len(df) * 100)}%): {repr(row['prompt'])}")
            print("".join(["-" for _ in range(100)]))
            print(f"Response A: {repr(row['response_a'])}")
            print("".join(["-" for _ in range(100)]))
            print(f"Response B: {repr(row['response_b'])}")
            label = int(input("1 for prompt A, 2 for prompt B. If both responses bad 3 for prompt A, 4 for prompt B: "))
            if label < 3:
                f.write(f"{label-1},0\n")
            else:
                f.write(f"{label-3},1\n")
            print("".join(["-" for _ in range(100)]))
            print(f"Chose {label}")
