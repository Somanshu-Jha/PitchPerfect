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
        detected_fillers = []

        for word in words:
            if word in self.fillers:
                count += 1
                detected_fillers.append(word)

        return detected_fillers