class FillerDetectionService:

    def __init__(self):
        """
        #---- Common filler words list
        """
        self.fillers = ["um", "uh", "like", "you know", "actually"]

    def detect(self, text: str):
        """
        #---- Detect filler words in transcript
        """
        words = text.lower().split()

        count = 0
        positions = []

        for i, word in enumerate(words):
            if word in self.fillers:
                count += 1
                positions.append(i)

        return {
            "filler_count": count,
            "positions": positions
        }