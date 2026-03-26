from concurrent.futures import ThreadPoolExecutor

from backend.services.transcription_service import TranscriptionService
from backend.services.correction_service import CorrectionService
from backend.services.semantic_service import SemanticService
from backend.services.audio_analysis_service import AudioAnalysisService
from backend.services.scoring_service import ScoringService
from backend.services.audio_preprocessing_service import AudioPreprocessingService
from backend.services.filler_detection_service import FillerDetectionService
from backend.services.feedback_service import FeedbackService
from backend.services.completeness_service import CompletenessService
import os

os.environ["OMP_NUM_THREADS"] = "8"


class SpeechPipeline:

    def __init__(self):
        self.transcriber = TranscriptionService()
        self.corrector = CorrectionService()
        self.semantic = SemanticService()
        self.audio_analyzer = AudioAnalysisService()
        self.scorer = ScoringService()
        self.preprocessor = AudioPreprocessingService()
        self.filler_service = FillerDetectionService()
        self.feedback_service = FeedbackService()
        self.completeness_service = CompletenessService()

    def process(self, audio_path: str):

        print("\n========== PIPELINE START ==========")

        # ---- STEP 1: PREPROCESS
        clean_audio = self.preprocessor.process(audio_path, "clean.wav")

        # ---- STEP 2: PARALLEL EXECUTION
        with ThreadPoolExecutor(max_workers=4) as executor:

            results = list(executor.map(
                lambda fn: fn(),
                [
                    lambda: self.transcriber.transcribe(clean_audio),
                    lambda: self.audio_analyzer.extract(clean_audio)
                ]
            ))

        raw_text, audio_features = results

        print(f"[RAW TRANSCRIPT]: {raw_text}")

        # ---- STEP 3: CORRECTION
        refined_text = self.corrector.refine(raw_text)

        # ---- STEP 4: SEMANTIC (SAFE)
        try:
            semantic_result = self.semantic.analyze(refined_text)
        except Exception as e:
            print(f"❌ Semantic failure: {e}")
            # Keep the correct shape so scoring+feedback services don't break
            from backend.nlp.entity_extractor import EntityExtractor
            try:
                structured = EntityExtractor().extract(refined_text)
            except Exception:
                structured = {}
            semantic_result = {
                "intent": {"detected": [], "confidence": 0.0},
                "structured": structured
            }

        # ---- STEP 5: COMPLETENESS (SAFE)
        try:
            completeness_issues = self.completeness_service.check(
                refined_text,
                semantic_result
            )
            
        except Exception as e:
            print(f"❌ Completeness failure: {e}")
            completeness_issues = []

        # ---- STEP 6: FILLER DETECTION
        try:
            fillers = self.filler_service.detect(refined_text)
        except Exception as e:
            print(f"❌ Filler detection failure: {e}")
            fillers = []

        # ---- STEP 7: SCORING (SAFE FALLBACK)
        try:
            scores = self.scorer.evaluate(
                refined_text,
                semantic_result,
                audio_features,
                fillers
            )
        except Exception as e:
            print("⚠️ Scoring fallback:", e)
            scores = {
                "overall_score": 6,
                "note": "Fallback scoring used"
            }

        # ---- STEP 8: FEEDBACK (SAFE)
        try:
            feedback = self.feedback_service.generate(
                refined_text,
                semantic_result,
                scores,
                fillers
            )
        except Exception as e:
            print(f"❌ Feedback failure: {e}")
            feedback = ["Feedback generation failed"]

        print(f"[SEMANTIC]: {semantic_result}")
        print(f"[SCORES]: {scores}")
        print(f"[FEEDBACK]: {feedback}")

        print("========== PIPELINE END ==========")

        return {
            "raw_transcript": raw_text,
            "refined_transcript": refined_text,
            "semantic": semantic_result,
            "audio_features": audio_features,
            "fillers": fillers,
            "scores": scores,
            "feedback": feedback,
            "completeness_issues": completeness_issues
        }