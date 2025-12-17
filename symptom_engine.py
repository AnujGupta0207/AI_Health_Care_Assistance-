import pandas as pd

# Load the CSV file
df = pd.read_csv("symptoms_data.csv")

# Preprocess: convert symptom strings to lists
df["symptom_list"] = df["symptoms"].apply(
    lambda x: [s.strip().lower() for s in str(x).split(";")]
)

def predict_diseases(user_text, top_k=3):
    """
    Very simple symptom matcher:
    Input: user_text like "I have fever, headache and body pain"
    Output: list of (disease, matched_symptom_count)
    """
    user_text = user_text.lower()

    scores = []
    for _, row in df.iterrows():
        disease = row["disease"]
        disease_symptoms = row["symptom_list"]

        match_count = 0
        for sym in disease_symptoms:
            if sym in user_text:
                match_count += 1

        if match_count > 0:
            scores.append((disease, match_count))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
