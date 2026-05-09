import pandas as pd
import numpy as np

def generate_aml_dataset(n_samples=50000, random_state=42):
    np.random.seed(random_state)

    n_normal     = int(n_samples * 0.98)
    n_laundering = n_samples - n_normal

    # Normal transactions
    normal = pd.DataFrame({
        "transaction_id":   range(n_normal),
        "sender_account":   np.random.randint(1000, 9999, n_normal),
        "receiver_account": np.random.randint(1000, 9999, n_normal),
        "amount":           np.random.exponential(500, n_normal).round(2),
        "transaction_type": np.random.choice(["transfer", "payment", "deposit", "withdrawal"], n_normal),
        "hour":             np.random.randint(8, 20, n_normal),
        "day_of_week":      np.random.randint(0, 5, n_normal),
        "num_transactions_sender":   np.random.randint(1, 10, n_normal),
        "num_transactions_receiver": np.random.randint(1, 10, n_normal),
        "same_bank":        np.random.choice([0, 1], n_normal, p=[0.3, 0.7]),
        "international":    np.random.choice([0, 1], n_normal, p=[0.9, 0.1]),
        "is_laundering":    0
    })

    # Suspicious transactions - fixed amount generation
    half = n_laundering // 2
    amounts = np.concatenate([
        np.random.uniform(9000, 9999, half),
        np.random.uniform(50000, 500000, n_laundering - half)
    ])

    laundering = pd.DataFrame({
        "transaction_id":   range(n_normal, n_samples),
        "sender_account":   np.random.randint(1000, 9999, n_laundering),
        "receiver_account": np.random.randint(1000, 9999, n_laundering),
        "amount":           amounts.round(2),
        "transaction_type": np.random.choice(["transfer", "payment"], n_laundering),
        "hour":             np.random.randint(0, 6, n_laundering),
        "day_of_week":      np.random.randint(5, 7, n_laundering),
        "num_transactions_sender":   np.random.randint(20, 100, n_laundering),
        "num_transactions_receiver": np.random.randint(20, 100, n_laundering),
        "same_bank":        np.random.choice([0, 1], n_laundering, p=[0.7, 0.3]),
        "international":    np.random.choice([0, 1], n_laundering, p=[0.3, 0.7]),
        "is_laundering":    1
    })

    df = pd.concat([normal, laundering], ignore_index=True).sample(frac=1, random_state=random_state)
    df["amount_log"] = np.log1p(df["amount"])
    df = df.reset_index(drop=True)
    return df
