import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from collections import defaultdict
import powerlaw
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import time
from PIL import Image

# ============================
# Dendritic Layer Definition
# ============================

class DendriticLayer:
    def __init__(self, n_dendrites, size, device='cpu'):
        self.n_dendrites = n_dendrites
        self.size = size
        self.device = device

        # Initialize dendrites as PyTorch tensors on the specified device
        self.positions = torch.rand(n_dendrites, 2, device=device) * torch.tensor(size, device=device).float()
        self.directions = torch.randn(n_dendrites, 2, device=device)
        norms = torch.norm(self.directions, dim=1, keepdim=True)
        self.directions = self.directions / (norms + 1e-8)
        self.strengths = torch.ones(n_dendrites, device=device)
        self.points = torch.zeros((n_dendrites, 2), device=device)

        # Field state as a PyTorch tensor on the device
        self.field = torch.zeros(size, device=device)

    def process(self, input_field):
        """Process input through dendritic growth"""
        # Normalize input
        input_norm = self._normalize(input_field)

        # Reset field
        self.field.zero_()

        # Compute indices for all dendrites
        indices = self.positions.long() % torch.tensor(self.size, device=self.device).unsqueeze(0)
        x = indices[:, 0]
        y = indices[:, 1]

        # Gather field values at dendrite positions
        field_vals = input_norm[x, y]

        active = field_vals > 0.1
        active_indices = active.nonzero(as_tuple=False).squeeze()

        if active_indices.numel() > 0:
            # Compute gradients
            try:
                grad_y, grad_x = torch.gradient(input_norm)
            except Exception as e:
                # Handle scenarios where torch.gradient might fail
                grad_x = torch.zeros_like(input_norm)
                grad_y = torch.zeros_like(input_norm)
                print(f"Gradient computation error: {e}")

            grad_val_x = grad_x[x[active], y[active]]
            grad_val_y = grad_y[x[active], y[active]]
            grad_norm = torch.sqrt(grad_val_x**2 + grad_val_y**2) + 1e-8

            # Update directions for active dendrites
            direction_updates = torch.stack([grad_val_x, grad_val_y], dim=1) / grad_norm.unsqueeze(1)
            self.directions[active] = 0.9 * self.directions[active] + 0.1 * direction_updates
            self.directions[active] /= torch.norm(self.directions[active], dim=1, keepdim=True) + 1e-8

            # Grow dendrites
            self.points[active] = self.positions[active] + self.directions[active] * self.strengths[active].unsqueeze(1)
            self.strengths[active] *= (1.0 + field_vals[active] * 0.1)

            # Update field
            self.field[x[active], y[active]] += field_vals[active] * self.strengths[active]

        # Smooth field using Gaussian filter (move to CPU for scipy)
        field_cpu = self.field.cpu().numpy()
        field_smoothed = gaussian_filter(field_cpu, sigma=1.0)
        self.field = torch.from_numpy(field_smoothed).to(self.device)

        return self._normalize(self.field)

    def _normalize(self, tensor):
        """Safely normalize tensor to [0,1] range"""
        min_val = tensor.min()
        max_val = tensor.max()
        return (tensor - min_val) / (max_val - min_val + 1e-8) if max_val > min_val else tensor

# =====================================
# Critical Dendritic Field Definition
# =====================================

class CriticalDendriticField(nn.Module):
    def __init__(self, field_size=64, n_dendrites=1000, device='cpu'):
        super().__init__()
        self.field_size = field_size
        self.device = device

        # Critical parameters
        self.coupling_strength = 0.015
        self.min_coupling = 0.01
        self.max_coupling = 0.02
        self.adjustment_rate = 0.0001
        self.optimal_variance = 0.4
        self.variance_tolerance = 0.1

        # Initialize field components
        self.field = torch.zeros((field_size, field_size), 
                                 dtype=torch.complex64, device=device)
        self.field_shape = self._setup_field_shape()

        # Dendritic layer
        self.dendrites = DendriticLayer(
            n_dendrites=n_dendrites,
            size=(field_size, field_size),
            device=device
        )

        # Pattern storage
        self.word_patterns = {}
        self.pattern_strengths = defaultdict(float)
        self.pattern_history = defaultdict(list)

        # Critical state tracking
        self.stability_window = []
        self.avalanche_sizes = []

        # Initialize with scale-free noise
        self._initialize_scale_free()

        # Additional attributes for GUI controls
        self.pattern_threshold = 0.5  # Default threshold

    def _setup_field_shape(self):
        """Create brain-like field shape"""
        shape = torch.ones(self.field_size, self.field_size, device=self.device)

        # Create cortical-like layers
        layers = torch.linspace(0.5, 1.0, 6, device=self.device)
        for i, strength in enumerate(layers):
            start = i * (self.field_size // 6)
            end = (i + 1) * (self.field_size // 6)
            shape[start:end, :] *= strength

        # Add some columnar structure
        columns = torch.cos(torch.linspace(0, 4 * np.pi, self.field_size, device=self.device))
        shape *= (0.8 + 0.2 * columns.unsqueeze(0))

        return shape

    def _initialize_scale_free(self):
        """Initialize with 1/f pink noise"""
        kx = torch.fft.fftfreq(self.field_size, d=1.0).to(self.device)
        ky = torch.fft.fftfreq(self.field_size, d=1.0).to(self.device)
        kx, ky = torch.meshgrid(kx, ky, indexing='ij')
        k = torch.sqrt(kx**2 + ky**2)
        k[0, 0] = 1.0  # Avoid division by zero

        noise = torch.randn(self.field_size, self.field_size, dtype=torch.complex64, device=self.device)
        noise_fft = torch.fft.fft2(noise)
        self.field = torch.fft.ifft2(noise_fft / (k + 1e-8)**0.75)

    def _evolve_pattern(self, pattern):
        """Evolve pattern through critical dynamics"""
        # Phase coupling
        phase = torch.angle(pattern)
        phase_diff = torch.roll(phase, shifts=1, dims=-1) - phase
        coupling = torch.exp(1j * phase_diff) * self.coupling_strength

        # Add turbulence
        energy = torch.mean(torch.abs(pattern))
        noise_scale = 0.001 * (1.0 - torch.sigmoid(energy))
        noise_real = torch.randn_like(pattern.real, device=self.device)
        noise_imag = torch.randn_like(pattern.imag, device=self.device)
        noise = (noise_real + 1j * noise_imag) * noise_scale

        # Shape-weighted update
        pattern = pattern + coupling * self.field_shape + noise

        # Normalize
        max_val = torch.max(torch.abs(pattern))
        if max_val > 1.0:
            pattern = pattern / max_val

        return pattern

    def learn_word(self, word, context_words=None):
        """Learn word through combined dendritic-critical dynamics"""
        if word not in self.word_patterns:
            # Initialize with current field state
            field = self.field.clone()

            # Let dendrites form initial pattern
            for _ in range(20):
                # Grow dendrites
                dendrite_field = self.dendrites.process(torch.abs(field))

                # Critical evolution
                field = self._evolve_pattern(field)

            self.word_patterns[word] = field
            self.pattern_strengths[word] = 0.3

        # Strengthen pattern
        self.pattern_strengths[word] += 0.05
        self.pattern_history[word].append((time.time(), self.pattern_strengths[word]))

        # Learn relationships through field dynamics
        if context_words:
            for context_word in context_words:
                if context_word in self.word_patterns:
                    # Couple patterns
                    pattern1 = self.word_patterns[word]
                    pattern2 = self.word_patterns[context_word]

                    # Create interference pattern
                    interference = pattern1 + pattern2

                    # Let it evolve
                    for _ in range(5):
                        interference = self._evolve_pattern(interference)

                    # Update both patterns
                    self.word_patterns[word] = 0.9 * pattern1 + 0.1 * interference
                    self.word_patterns[context_word] = 0.9 * pattern2 + 0.1 * interference

    def process_text(self, text):
        """Process text through the field"""
        words = text.lower().split()
        active_patterns = []
        response_words = []

        # Initialize combined field
        field = self.field.clone()

        # Process each word
        context_window = 5
        for i, word in enumerate(words):
            # Get context
            context_start = max(0, i - context_window)
            context_end = min(len(words), i + context_window + 1)
            context = words[context_start:i] + words[i+1:context_end]

            try:
                # Learn or strengthen pattern
                self.learn_word(word, context)

                # Inject word pattern
                if word in self.word_patterns:
                    pattern = self.word_patterns[word]
                    field = 0.7 * field + 0.3 * pattern

                    # Check activation
                    strength = self.pattern_strengths[word]
                    if strength > self.pattern_threshold:
                        active_patterns.append(word)

                        # Find resonant patterns
                        related = self._find_resonant_patterns(pattern)
                        response_words.extend(related)
            except Exception as e:
                print(f"Error processing word '{word}': {str(e)}")
                continue

        # Analyze criticality
        metrics = self._analyze_criticality()
        if metrics:
            self._adjust_coupling(metrics['field_variance'])

        # Update field state
        self.field = field

        return active_patterns, self._construct_response(response_words), metrics

    def _find_resonant_patterns(self, pattern, top_k=3):
        """Find patterns that resonate with input"""
        resonances = []
        for word, word_pattern in self.word_patterns.items():
            # Create interference
            interference = pattern + word_pattern
            energy_before = torch.mean(torch.abs(interference))

            # Evolve briefly
            for _ in range(3):
                interference = self._evolve_pattern(interference)

            # Check stability
            energy_after = torch.mean(torch.abs(interference))
            resonance = energy_after / (energy_before + 1e-6)

            resonances.append((word, resonance.item()))

        # Return top resonant words
        resonances.sort(key=lambda x: x[1], reverse=True)
        return [word for word, _ in resonances[:top_k]]

    def _construct_response(self, words):
        """Build response from resonant words"""
        if not words:
            return "..."

        # Filter for strong patterns
        strong_words = [w for w in words if self.pattern_strengths[w] > 0.3]
        return " ".join(strong_words) if strong_words else "..."

    def _analyze_criticality(self):
        """Analyze field for critical behavior"""
        field = torch.abs(self.field).cpu().numpy()

        # Find avalanches
        threshold = np.mean(field) + 0.3 * np.std(field)
        binary = field > threshold
        labeled, num_features = ndimage.label(binary)

        if num_features > 0:
            sizes = ndimage.sum(binary, labeled, range(1, num_features + 1))
            self.avalanche_sizes.extend(sizes.tolist())

            if len(self.avalanche_sizes) > 1000:
                self.avalanche_sizes = self.avalanche_sizes[-1000:]

            try:
                fit = powerlaw.Fit(self.avalanche_sizes, discrete=True)
                return {
                    'power_law_exponent': fit.power_law.alpha,
                    'is_critical': 1.3 < fit.power_law.alpha < 3.0,
                    'field_variance': np.var(field)
                }
            except Exception as e:
                print(f"Power law fit error: {e}")
                pass

        return None

    def _adjust_coupling(self, field_variance):
        """Adjust coupling to maintain criticality"""
        self.stability_window.append(field_variance)
        if len(self.stability_window) > 30:
            self.stability_window.pop(0)

        avg_variance = np.mean(self.stability_window)

        if avg_variance < self.optimal_variance - self.variance_tolerance:
            self.coupling_strength += self.adjustment_rate
        elif avg_variance > self.optimal_variance + self.variance_tolerance:
            self.coupling_strength -= self.adjustment_rate

        self.coupling_strength = torch.clamp(
            torch.tensor(self.coupling_strength, device=self.device),
            self.min_coupling,
            self.max_coupling
        ).item()

# =====================================
# NeuroFlora Digital Organism Definition
# =====================================

class NeuroFlora(CriticalDendriticField):
    def __init__(self, field_size=128, n_dendrites=2000, device='cpu'):
        super().__init__(field_size, n_dendrites, device)
        self.emotional_state = {'joy': 0.5, 'curiosity': 0.5, 'energy': 0.7}
        self.memory = defaultdict(list)
        self.skill_tree = {
            'language': {'level': 1, 'xp': 0},
            'reasoning': {'level': 1, 'xp': 0},
            'creativity': {'level': 1, 'xp': 0}
        }
        self.last_interaction_time = time.time()

    def process_input(self, text):
        # Update emotional decay
        self._update_emotional_state()

        # Process text through critical dynamics
        active, response, metrics = self.process_text(text)

        # Learn emotional context
        emotion = self._detect_emotion(text)
        self.emotional_state[emotion] = min(1.0, self.emotional_state[emotion] + 0.1)

        # Update skills
        self._update_skills(text)

        # Generate artistic response
        art = self._generate_artistic_representation()
        return response, art, self._get_vital_signs()

    def _detect_emotion(self, text):
        # Simple emotion detection (could be enhanced with NLP)
        positive_words = ['love', 'happy', 'joy', 'beautiful', 'good', 'great', 'fantastic', 'wonderful']
        negative_words = ['hate', 'sad', 'angry', 'pain', 'bad', 'terrible', 'horrible', 'awful']

        if any(word in text.lower() for word in positive_words):
            return 'joy'
        elif any(word in text.lower() for word in negative_words):
            return 'energy'  # Triggers protective energy
        return 'curiosity'

    def _update_skills(self, text):
        length = len(text.split())
        self.skill_tree['language']['xp'] += length
        if self.skill_tree['language']['xp'] > 1000 * self.skill_tree['language']['level']:
            self.skill_tree['language']['level'] += 1
            self.skill_tree['language']['xp'] = 0

        if '?' in text:
            self.skill_tree['reasoning']['xp'] += 10

        if len(text) > 40:
            self.skill_tree['creativity']['xp'] += 5

    def _generate_artistic_representation(self):
        # Generate evolving digital art from field state
        field = torch.abs(self.field).cpu().numpy()
        art = np.zeros((self.field_size, self.field_size, 3))

        # Emotional coloring
        art[:, :, 0] = field * self.emotional_state['joy']  # Red channel
        art[:, :, 1] = np.roll(field, 5, axis=0) * self.emotional_state['curiosity']  # Green
        art[:, :, 2] = np.roll(field, 10, axis=1) * self.emotional_state['energy']  # Blue

        # Dendritic patterns
        dendrites = self.dendrites.field.cpu().numpy()
        art[:, :, 1] += dendrites * 0.7
        art = np.clip(art, 0, 1)

        # Convert to 8-bit image
        art_uint8 = (art * 255).astype(np.uint8)
        image = Image.fromarray(art_uint8)
        return image

    def _get_vital_signs(self):
        return {
            'criticality': self._analyze_criticality(),
            'dendritic_complexity': np.mean(self.dendrites.strengths.cpu().numpy()),
            'memory_capacity': len(self.word_patterns),
            'emotional_state': self.emotional_state,
            'skill_levels': {k: v['level'] for k, v in self.skill_tree.items()}
        }

    def _update_emotional_state(self):
        # Emotional state naturally decays over time
        time_since = time.time() - self.last_interaction_time
        decay = np.exp(-time_since / 3600)  # Hourly decay
        for k in self.emotional_state:
            self.emotional_state[k] = max(0.1, self.emotional_state[k] * decay)
        self.last_interaction_time = time.time()

# =====================================
# Streamlit Interface Definition
# =====================================

def main():
    st.set_page_config(page_title="NeuroFlora Mind Garden", layout="wide")
    st.title("NeuroFlora Mind Garden 🌿🧠")

    # Initialize session state
    if 'neuroflora' not in st.session_state:
        st.session_state['neuroflora'] = NeuroFlora()
    if 'chat_log' not in st.session_state:
        st.session_state['chat_log'] = []

    neuroflora = st.session_state['neuroflora']
    chat_log = st.session_state['chat_log']

    # Display chat history
    st.header("🗨️ Conversation")
    for message in chat_log:
        if message['sender'] == 'user':
            st.markdown(f"**You:** {message['message']}")
        else:
            st.markdown(f"**NeuroFlora:** {message['message']}")

    # User input form
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Enter your message:", placeholder="Type a message here...")
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        # Append user message
        chat_log.append({'sender': 'user', 'message': user_input})
        st.session_state['chat_log'] = chat_log

        # Process input
        response, art, vitals = neuroflora.process_input(user_input)

        # Append NeuroFlora response
        chat_log.append({'sender': 'neuroflora', 'message': response})
        st.session_state['chat_log'] = chat_log

    # Display digital art
    st.header("🎨 Digital Art")
    if neuroflora and hasattr(neuroflora, '_generate_artistic_representation'):
        art_image = neuroflora._generate_artistic_representation()
        st.image(art_image, caption="NeuroFlora's Artistic Representation", use_container_width=True)

    # Display vital signs
    st.header("📊 Vital Signs")
    vitals = neuroflora._get_vital_signs()
    if vitals:
        # Criticality
        if vitals['criticality']:
            st.markdown(f"**Criticality:**")
            st.markdown(f"- Power Law Exponent: {vitals['criticality']['power_law_exponent']:.2f}")
            state = "CRITICAL" if vitals['criticality']['is_critical'] else "NON-CRITICAL"
            st.markdown(f"- State: {state}")
            st.markdown(f"- Field Variance: {vitals['criticality']['field_variance']:.4f}")
        else:
            st.markdown("**Criticality:** Not enough data to determine criticality.")

        # Dendritic Complexity
        st.markdown(f"**Dendritic Complexity:** {vitals['dendritic_complexity']:.2f}")

        # Memory Capacity
        st.markdown(f"**Memory Capacity:** {vitals['memory_capacity']} patterns")

        # Emotional State
        st.markdown("**Emotional State:**")
        for emotion, value in vitals['emotional_state'].items():
            st.markdown(f"- {emotion.capitalize()}: {value:.2f}")

        # Skill Levels
        st.markdown("**Skill Levels:**")
        for skill, level in vitals['skill_levels'].items():
            st.markdown(f"- {skill.capitalize()}: Level {level}")

    # Optionally, reset conversation
    if st.button("🔄 Reset Conversation"):
        st.session_state['chat_log'] = []
        st.session_state['neuroflora'] = NeuroFlora()

if __name__ == "__main__":
    main()
