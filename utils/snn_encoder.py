"""
SNN (Spiking Neural Network) Encoder for Temporal Feature Extraction

This module provides a hybrid neural network approach combining:
1. SNN encoder for temporal dynamics and biological plausibility
2. VLM integration for high-level reasoning

Key Features:
- Rate-based spike encoding from game frames
- Leaky Integrate-and-Fire (LIF) neurons for temporal processing
- Spike frequency analysis for detecting changes/events
- Integration with existing VLM pipeline

Novel Contributions:
- Temporal dynamics: SNNs naturally capture frame-to-frame changes
- Energy efficiency: Spike-based computation is more efficient than frame processing
- Biological plausibility: Mimics neural computation principles
- Event detection: Spike patterns can detect important game events (battles, dialogues, transitions)

References:
- snnTorch: Neuromorphic computing in PyTorch
- SpikingJelly: Deep learning with spiking neural networks
- Tavanaei et al. 2019: Deep learning in spiking neural networks
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from collections import deque
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class SNNConfig:
    """Configuration for SNN encoder"""
    # Frame processing
    downsample_size: Tuple[int, int] = (40, 40)  # Downsample frames for SNN processing
    temporal_window: int = 4  # Number of frames to process together
    
    # LIF neuron parameters
    v_threshold: float = 1.0  # Spike threshold
    v_reset: float = 0.0  # Reset potential after spike
    tau_mem: float = 0.5  # Membrane time constant
    tau_syn: float = 0.3  # Synaptic time constant
    
    # Spike encoding
    encoding_method: str = "rate"  # "rate" or "temporal"
    max_spike_rate: float = 1.0  # Maximum spike rate for rate encoding
    
    # Feature extraction
    spike_window_size: int = 10  # Window for spike frequency analysis
    change_detection_threshold: float = 0.3  # Threshold for detecting significant changes


class LIFNeuron:
    """
    Leaky Integrate-and-Fire (LIF) neuron model
    
    Biologically-inspired neuron that:
    1. Integrates input current over time
    2. Leaks charge exponentially
    3. Fires a spike when threshold is reached
    """
    def __init__(self, v_threshold=1.0, v_reset=0.0, tau_mem=0.5):
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.tau_mem = tau_mem
        self.v_mem = v_reset  # Current membrane potential
        self.spike_history = deque(maxlen=100)
        
    def step(self, input_current: float, dt: float = 0.1) -> bool:
        """
        Update neuron state and check for spike
        
        Args:
            input_current: Input current to neuron
            dt: Time step
            
        Returns:
            bool: True if neuron spiked, False otherwise
        """
        # Leaky integration: dv/dt = -v/tau + I
        self.v_mem += dt * ((-self.v_mem / self.tau_mem) + input_current)
        
        # Check for spike
        if self.v_mem >= self.v_threshold:
            self.v_mem = self.v_reset
            self.spike_history.append(time.time())
            return True
        
        return False
    
    def get_spike_rate(self, window_seconds: float = 1.0) -> float:
        """Calculate recent spike rate"""
        current_time = time.time()
        recent_spikes = [t for t in self.spike_history if current_time - t < window_seconds]
        return len(recent_spikes) / window_seconds if window_seconds > 0 else 0.0


class SNNEncoder:
    """
    Spiking Neural Network encoder for game frames
    
    This encoder:
    1. Converts frames to spike trains (rate-based or temporal encoding)
    2. Processes spikes through LIF neuron layers
    3. Extracts temporal features (spike rates, patterns, changes)
    4. Provides features to augment VLM input
    """
    
    def __init__(self, config: SNNConfig = None):
        self.config = config or SNNConfig()
        
        # Frame history for temporal processing
        self.frame_history = deque(maxlen=self.config.temporal_window)
        
        # SNN layers: Input → Hidden → Output
        # Each layer has neurons for different regions/features
        num_input_neurons = self.config.downsample_size[0] * self.config.downsample_size[1]
        num_hidden_neurons = 256
        num_output_neurons = 64
        
        self.hidden_layer = [LIFNeuron(
            v_threshold=self.config.v_threshold,
            v_reset=self.config.v_reset,
            tau_mem=self.config.tau_mem
        ) for _ in range(num_hidden_neurons)]
        
        self.output_layer = [LIFNeuron(
            v_threshold=self.config.v_threshold,
            v_reset=self.config.v_reset,
            tau_mem=self.config.tau_mem
        ) for _ in range(num_output_neurons)]
        
        # Random weight initialization (simplified - in practice use trained weights)
        self.w_input_hidden = np.random.randn(num_input_neurons, num_hidden_neurons) * 0.1
        self.w_hidden_output = np.random.randn(num_hidden_neurons, num_output_neurons) * 0.1
        
        # Feature tracking
        self.temporal_features = {
            "spike_rate": 0.0,
            "change_detected": False,
            "activity_level": 0.0,
            "temporal_contrast": 0.0,
        }
        
        logger.info(f"SNNEncoder initialized: {num_hidden_neurons} hidden, {num_output_neurons} output neurons")
    
    def encode_frame_to_spikes(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert frame to spike train using rate-based encoding
        
        Args:
            frame: RGB frame (H, W, 3)
            
        Returns:
            spike_train: Binary spike array (flattened frame size)
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            frame_gray = np.mean(frame, axis=2)
        else:
            frame_gray = frame
        
        # Downsample for efficiency
        from PIL import Image
        frame_img = Image.fromarray(frame_gray.astype(np.uint8))
        frame_small = frame_img.resize(self.config.downsample_size, Image.Resampling.LANCZOS)
        frame_small = np.array(frame_small).astype(np.float32) / 255.0
        
        # Rate-based encoding: pixel intensity → spike probability
        # Brighter pixels = higher spike rate
        if self.config.encoding_method == "rate":
            spike_probs = frame_small.flatten() * self.config.max_spike_rate
            spike_train = (np.random.random(spike_probs.shape) < spike_probs).astype(np.float32)
        else:
            # Temporal encoding: pixel intensity → spike timing (future work)
            spike_train = (frame_small.flatten() > 0.5).astype(np.float32)
        
        return spike_train
    
    def process_spikes(self, spike_train: np.ndarray, dt: float = 0.1) -> Dict[str, float]:
        """
        Process spike train through SNN layers
        
        Args:
            spike_train: Input spikes (flattened frame)
            dt: Time step
            
        Returns:
            features: Extracted temporal features
        """
        # Hidden layer processing
        hidden_currents = np.dot(spike_train, self.w_input_hidden)
        hidden_spikes = np.array([
            neuron.step(current, dt) for neuron, current in zip(self.hidden_layer, hidden_currents)
        ], dtype=np.float32)
        
        # Output layer processing
        output_currents = np.dot(hidden_spikes, self.w_hidden_output)
        output_spikes = np.array([
            neuron.step(current, dt) for neuron, current in zip(self.output_layer, output_currents)
        ], dtype=np.float32)
        
        # Extract features
        features = {
            "spike_rate": np.mean([n.get_spike_rate(1.0) for n in self.output_layer]),
            "activity_level": np.sum(output_spikes) / len(output_spikes),
            "hidden_activity": np.sum(hidden_spikes) / len(hidden_spikes),
        }
        
        return features
    
    def encode_frame(self, frame: np.ndarray) -> Dict[str, any]:
        """
        Main encoding function: Frame → Spikes → Features
        
        Args:
            frame: RGB game frame
            
        Returns:
            encoding: Dictionary with spike features and temporal dynamics
        """
        # Store frame in history
        self.frame_history.append(frame.copy())
        
        # Encode current frame to spikes
        spike_train = self.encode_frame_to_spikes(frame)
        
        # Process through SNN
        features = self.process_spikes(spike_train)
        
        # Temporal change detection (compare with previous frames)
        if len(self.frame_history) >= 2:
            prev_frame = self.frame_history[-2]
            frame_diff = np.abs(frame.astype(float) - prev_frame.astype(float)).mean()
            temporal_contrast = frame_diff / 255.0
            features["temporal_contrast"] = temporal_contrast
            features["change_detected"] = temporal_contrast > self.config.change_detection_threshold
        else:
            features["temporal_contrast"] = 0.0
            features["change_detected"] = False
        
        # Update tracked features
        self.temporal_features.update(features)
        
        return {
            "spike_features": features,
            "spike_rate": features["spike_rate"],
            "change_detected": features["change_detected"],
            "activity_level": features["activity_level"],
            "temporal_contrast": features.get("temporal_contrast", 0.0),
            "temporal_window_size": len(self.frame_history),
        }
    
    def get_textual_summary(self) -> str:
        """
        Get human-readable summary of SNN features for VLM integration
        
        Returns:
            summary: Text description of temporal dynamics
        """
        features = self.temporal_features
        
        # Activity level interpretation
        if features["activity_level"] > 0.5:
            activity = "HIGH (screen changing rapidly)"
        elif features["activity_level"] > 0.2:
            activity = "MEDIUM (some movement)"
        else:
            activity = "LOW (static scene)"
        
        # Change detection
        change_status = "SIGNIFICANT CHANGE DETECTED" if features["change_detected"] else "Scene stable"
        
        summary = f"""
🧠 Temporal Dynamics (SNN Analysis):
- Neural Activity: {activity}
- Spike Rate: {features['spike_rate']:.3f} Hz
- Temporal Change: {features['temporal_contrast']:.3f}
- Change Detection: {change_status}
- Interpretation: {self._interpret_features(features)}
"""
        return summary.strip()
    
    def _interpret_features(self, features: Dict[str, float]) -> str:
        """Interpret SNN features for decision-making"""
        if features["change_detected"] and features["activity_level"] > 0.4:
            return "Scene transition or battle start likely"
        elif features["activity_level"] < 0.1 and features["temporal_contrast"] < 0.05:
            return "Static scene - may be stuck or in dialogue"
        elif features["spike_rate"] > 0.5:
            return "High temporal dynamics - active gameplay"
        else:
            return "Normal gameplay activity"
    
    def get_feature_vector(self) -> np.ndarray:
        """
        Get numerical feature vector for ML models
        
        Returns:
            features: Numpy array of extracted features
        """
        f = self.temporal_features
        return np.array([
            f["spike_rate"],
            f["activity_level"],
            f["temporal_contrast"],
            float(f["change_detected"]),
            len(self.frame_history) / self.config.temporal_window,  # Fill ratio
        ], dtype=np.float32)
    
    def reset(self):
        """Reset encoder state"""
        self.frame_history.clear()
        for neuron in self.hidden_layer + self.output_layer:
            neuron.v_mem = neuron.v_reset
            neuron.spike_history.clear()
        logger.info("SNNEncoder reset")


# Convenience function for integration
def create_snn_encoder(config: Dict = None) -> SNNEncoder:
    """
    Factory function to create SNN encoder with custom config
    
    Args:
        config: Dictionary with configuration parameters
        
    Returns:
        encoder: Configured SNNEncoder instance
    """
    if config:
        snn_config = SNNConfig(**config)
    else:
        snn_config = SNNConfig()
    
    return SNNEncoder(snn_config)


# Example integration with VLM
def augment_vlm_prompt_with_snn(base_prompt: str, snn_encoder: SNNEncoder, frame: np.ndarray) -> str:
    """
    Augment VLM prompt with SNN temporal features
    
    Args:
        base_prompt: Original VLM prompt
        snn_encoder: SNN encoder instance
        frame: Current game frame
        
    Returns:
        augmented_prompt: Prompt with SNN analysis added
    """
    # Encode frame and extract features
    encoding = snn_encoder.encode_frame(frame)
    snn_summary = snn_encoder.get_textual_summary()
    
    # Prepend SNN analysis to VLM prompt
    augmented_prompt = f"""{snn_summary}

---

{base_prompt}
"""
    
    return augmented_prompt


if __name__ == "__main__":
    # Example usage
    print("SNN Encoder Test")
    print("=" * 60)
    
    # Create encoder
    encoder = create_snn_encoder()
    
    # Simulate processing frames
    for i in range(5):
        # Create dummy frame (240x160x3)
        dummy_frame = np.random.randint(0, 255, (240, 160, 3), dtype=np.uint8)
        
        # Encode
        result = encoder.encode_frame(dummy_frame)
        
        print(f"\nFrame {i+1}:")
        print(f"  Spike Rate: {result['spike_rate']:.3f} Hz")
        print(f"  Activity: {result['activity_level']:.3f}")
        print(f"  Change Detected: {result['change_detected']}")
    
    # Get summary
    print("\n" + encoder.get_textual_summary())
    print("\nFeature Vector:", encoder.get_feature_vector())

