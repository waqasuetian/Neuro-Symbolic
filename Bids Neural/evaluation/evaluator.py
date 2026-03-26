import numpy as np
from typing import List, Dict
from config.settings import Config

class SeizureEvaluator:
    def __init__(self, config: Config):
        self.config = config
        
    def detect_events(self, preds: np.ndarray, timestamps: List) -> List[Dict]:
        """Convert model predictions to seizure events"""
        events = []
        current_event = None
        
        for i, (pred, ts) in enumerate(zip(preds, timestamps)):
            if pred == 1:  # Seizure class
                if current_event is None:
                    current_event = {
                        'start': ts['epoch_start_sec'],
                        'end': ts['epoch_end_sec'],
                        'confidence': [],
                        'file': ts['file']
                    }
                else:
                    current_event['end'] = ts['epoch_end_sec']
                current_event['confidence'].append(pred[1])
            elif current_event is not None:
                events.append(self._finalize_event(current_event))
                current_event = None
                
        return events
    
    def _finalize_event(self, event: Dict) -> Dict:
        """Calculate final event metrics"""
        duration = event['end'] - event['start']
        if duration >= self.config.data.min_seizure_duration:
            return {
                **event,
                'duration': duration,
                'confidence': np.mean(event['confidence'])
            }
        return None