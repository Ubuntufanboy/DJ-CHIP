import numpy as np
import wave
import mido
import pyaudio

class Channel:
    def __init__(self, instrument_name, bit_depth=16, sample_rate=44100, waveform='sine', volume=1.0):
        self.instrument_name = instrument_name  # real name of the instrument
        self.bit_depth = bit_depth
        self.sample_rate = sample_rate
        self.waveform = waveform.lower()
        self.volume = volume

    def generate_wave(self, frequency, duration, volume=None):
        vol = volume if volume is not None else self.volume
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        if self.waveform == 'sine':
            data = np.sin(2 * np.pi * frequency * t)
        elif self.waveform == 'square':
            data = np.sign(np.sin(2 * np.pi * frequency * t))
        elif self.waveform == 'triangle':
            data = 2 * np.arcsin(np.sin(2 * np.pi * frequency * t)) / np.pi
        elif self.waveform == 'sawtooth':
            data = 2 * (t * frequency - np.floor(0.5 + t * frequency))
        elif self.waveform == 'noise':
            num_samples = len(t)
            noise_data = np.zeros(num_samples)
            lfsr = 0x7FFF  # nonzero 15-bit initial state
            # Use the provided frequency as a rough clock divider for noise updates.
            step = max(1, int(self.sample_rate / frequency)) if frequency > 0 else 1
            for i in range(num_samples):
                if i % step == 0:
                    # Feedback from bit0 XOR bit1.
                    bit = (lfsr & 1) ^ ((lfsr >> 1) & 1)
                    lfsr = (lfsr >> 1) | (bit << 14)  # maintain 15 bits (bit14 is the top bit)
                # Output: if the LSB is 1, output 1.0; if 0, output -1.0.
                noise_data[i] = 1.0 if (lfsr & 1) else -1.0
            data = noise_data
        else:
            data = np.sin(2 * np.pi * frequency * t)
        return data * vol

def get_instrument_mapping(midi_file):
    mid = mido.MidiFile(midi_file)
    mapping = {}
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'program_change':
                if msg.channel not in mapping:
                    mapping[msg.channel] = msg.program
    instrument_map = {}
    used_channels = set()
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                used_channels.add(msg.channel)
    for ch in used_channels:
        if ch == 9:
            instrument_map[ch] = "Standard Drum Kit"
        elif ch in mapping:
            program = mapping[ch]
            if 0 <= program < len(GM_INSTRUMENTS):
                instrument_map[ch] = GM_INSTRUMENTS[program]
            else:
                instrument_map[ch] = "Unknown"
        else:
            instrument_map[ch] = "Unknown"
    return instrument_map

def midi_to_events_with_mapping(midi_file, channel_objects):
    mid = mido.MidiFile(midi_file)
    events = []
    current_time = 0.0
    default_duration = 0.5
    for msg in mid:
        current_time += msg.time
        if msg.type == 'note_on' and msg.velocity > 0:
            frequency = 440.0 * 2 ** ((msg.note - 69) / 12.0)
            if msg.channel in channel_objects:
                events.append((channel_objects[msg.channel], frequency, current_time, default_duration))
    return events

def mix_events(events, sample_rate, bit_depth):
    total_duration = max(start + duration for (_, _, start, duration) in events)
    total_samples = int(total_duration * sample_rate)
    mix = np.zeros(total_samples, dtype=np.float32)
    for ch, freq, start, duration in events:
        start_sample = int(start * sample_rate)
        wave_data = ch.generate_wave(freq, duration)
        end_sample = start_sample + len(wave_data)
        if end_sample > total_samples:
            wave_data = wave_data[:total_samples - start_sample]
            end_sample = total_samples
        mix[start_sample:end_sample] += wave_data
    max_val = np.max(np.abs(mix))
    if max_val > 1:
        mix = mix / max_val
    kernel_size = 100
    kernel = np.ones(kernel_size) / kernel_size
    smooth_mix = np.convolve(mix, kernel, mode='same')
    return smooth_mix

def write_wav(filename, data, sample_rate, bit_depth):
    if bit_depth == 16:
        data_int = (data * 32767).astype(np.int16)
        sampwidth = 2
    elif bit_depth == 8:
        data_int = ((data + 1) * 127.5).astype(np.uint8)
        sampwidth = 1
    else:
        raise ValueError("Must choose 8 or 16.")
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(data_int.tobytes())

def play_wav(filename):
    chunk = 1024
    wf = wave.open(filename, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(chunk)
    while data:
        stream.write(data)
        data = wf.readframes(chunk)
    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    print("=== WAV File Mixer Sound System ===")
    midi_file = input("Enter path to MIDI file: ")
    sample_rate = int(input("Enter sample rate (e.g., 44100): "))
    bit_depth = int(input("Enter bit depth (8 or 16): "))

    instrument_map = get_instrument_mapping(midi_file)
    print("\nDetected Instruments:")
    for ch, inst in instrument_map.items():
        print(f"  MIDI Channel {ch}: {inst}")

    channel_objects = {}
    for ch, inst in instrument_map.items():
        prompt_wave = f"Select waveform for '{inst}' (sine, square, triangle, sawtooth, noise): "
        waveform = input(prompt_wave)
        prompt_vol = f"Enter volume for '{inst}' (0.0 to 1.0): "
        volume = float(input(prompt_vol))
        channel_objects[ch] = Channel(instrument_name=inst,
                                      bit_depth=bit_depth,
                                      sample_rate=sample_rate,
                                      waveform=waveform,
                                      volume=volume)

    print("\nInstrument Mapping:")
    for ch, ch_obj in channel_objects.items():
        print(f"  MIDI Channel {ch}: {ch_obj.instrument_name} -> {ch_obj.waveform} at volume {ch_obj.volume}")

    print("\nProcessing MIDI file and converting to events...")
    events = midi_to_events_with_mapping(midi_file, channel_objects)
    print("Mixing events into a single audio buffer with smoothing...")
    mix = mix_events(events, sample_rate, bit_depth)

    output_wav = "output.wav"
    print("Writing mixed audio to WAV file...")
    write_wav(output_wav, mix, sample_rate, bit_depth)

    print("Playing WAV file...")
    play_wav(output_wav)
    print("Done.")

if __name__ == "__main__":
    main()
