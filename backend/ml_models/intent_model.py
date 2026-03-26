# -------------------- INTENT MODEL --------------------
from sklearn.linear_model import LogisticRegression

class IntentModel:

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, embedding):
        return self.model.predict([embedding])[0]