import os
import torch
import torchaudio
import sox
import json
import argparse
import csv
import numpy as np
import soundfile as sf

from text_normalization import text_normalize
from align_utils import (
    get_uroman_tokens,
    time_to_frame,
    load_model_dict,
    merge_repeats,
    get_spans,
)
import torchaudio.functional as F

SAMPLING_FREQ = 16000
EMISSION_INTERVAL = 30
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paramètres pour la détection de parole
THRESHOLD = 0.02      # Seuil pour détecter le début de la parole (à ajuster si nécessaire)
SILENCE_DURATION = 0.2  # Durée de silence à ajouter en secondes (200 ms)

def detect_speech(audio, sr, threshold=THRESHOLD):
    """
    Détecte le début et la fin de la parole dans un signal audio.
    Retourne les indices (en échantillons) correspondant au premier et dernier point
    où l'énergie dépasse le seuil.
    """
    energy = np.abs(audio)
    speech_indices = np.where(energy > threshold)[0]
    if len(speech_indices) == 0:
        return 0, len(audio)
    start = speech_indices[0]
    end = speech_indices[-1]
    return start, end

def generate_emissions(model, audio_file):
    waveform, _ = torchaudio.load(audio_file)  # waveform: channels x T
    waveform = waveform.to(DEVICE)
    total_duration = sox.file_info.duration(audio_file)

    audio_sf = sox.file_info.sample_rate(audio_file)
    assert audio_sf == SAMPLING_FREQ

    emissions_arr = []
    with torch.inference_mode():
        i = 0
        while i < total_duration:
            segment_start_time, segment_end_time = (i, i + EMISSION_INTERVAL)

            context = EMISSION_INTERVAL * 0.1
            input_start_time = max(segment_start_time - context, 0)
            input_end_time = min(segment_end_time + context, total_duration)
            waveform_split = waveform[
                :,
                int(SAMPLING_FREQ * input_start_time) : int(
                    SAMPLING_FREQ * (input_end_time)
                ),
            ]

            model_outs, _ = model(waveform_split)
            emissions_ = model_outs[0]
            emission_start_frame = time_to_frame(segment_start_time)
            emission_end_frame = time_to_frame(segment_end_time)
            offset = time_to_frame(input_start_time)

            emissions_ = emissions_[
                emission_start_frame - offset : emission_end_frame - offset, :
            ]
            emissions_arr.append(emissions_)
            i += EMISSION_INTERVAL

    emissions = torch.cat(emissions_arr, dim=0).squeeze()
    emissions = torch.log_softmax(emissions, dim=-1)

    stride = float(waveform.size(1) * 1000 / emissions.size(0) / SAMPLING_FREQ)

    return emissions, stride

def get_alignments(
    audio_file,
    tokens,
    model,
    dictionary,
    use_star,
):
    # Generate emissions
    emissions, stride = generate_emissions(model, audio_file)
    T, N = emissions.size()
    if use_star:
        emissions = torch.cat([emissions, torch.zeros(T, 1).to(DEVICE)], dim=1)

    # Force Alignment
    if tokens:
        token_indices = [dictionary[c] for c in " ".join(tokens).split(" ") if c in dictionary]
    else:
        print(f"Empty transcript!!!!! for audio file {audio_file}")
        token_indices = []

    blank = dictionary["<blank>"]
    
    targets = torch.tensor(token_indices, dtype=torch.int32).to(DEVICE)
    
    input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1)
    target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1)
    path, _ = F.forced_align(
        emissions.unsqueeze(0), targets.unsqueeze(0), input_lengths, target_lengths, blank=blank
    )
    path = path.squeeze().to("cpu").tolist()
    
    segments = merge_repeats(path, {v: k for k, v in dictionary.items()})
    return segments, stride

def main(args):
    assert not os.path.exists(
        args.outdir
    ), f"Error: Output path exists already {args.outdir}"
    
    transcripts = []
    with open(args.text_filepath, encoding="utf-8") as f:
        transcripts = [line.strip() for line in f]
    print("Read {} lines from {}".format(len(transcripts), args.text_filepath))

    norm_transcripts = [text_normalize(line.strip(), args.lang) for line in transcripts]
    tokens = get_uroman_tokens(norm_transcripts, args.uroman_path, args.lang)

    model, dictionary = load_model_dict()
    model = model.to(DEVICE)
    if args.use_star:
        dictionary["<star>"] = len(dictionary)
        tokens = ["<star>"] + tokens
        transcripts = ["<star>"] + transcripts
        norm_transcripts = ["<star>"] + norm_transcripts

    segments, stride = get_alignments(
        args.audio_filepath,
        tokens,
        model,
        dictionary,
        args.use_star,
    )
    # Get spans of each line in input text file
    spans = get_spans(tokens, segments)

    os.makedirs(args.outdir)
    
    # Extraire le nom de fichier source sans extension
    nom_fichier_source = os.path.splitext(os.path.basename(args.audio_filepath))[0]

    # Calculer la durée totale de l'audio
    total_audio_duration = sox.file_info.duration(args.audio_filepath)

    manifest_path = f"{args.outdir}/manifest.json"
    metadata_path = f"{args.outdir}/metadata.csv"

    with open(manifest_path, "w", encoding="utf-8") as manifest_file, \
         open(metadata_path, "w", newline="", encoding="utf-8") as metadata_file:
        csv_writer = csv.writer(metadata_file, delimiter="|")

        for i, t in enumerate(transcripts):
            span = spans[i]
            seg_start_idx = span[0].start
            seg_end_idx = span[-1].end

            # Calculer les temps de début et fin en secondes pour le segment (avant traitement)
            audio_start_sec = seg_start_idx * stride / 1000
            audio_end_sec = seg_end_idx * stride / 1000 

            # Ajouter un buffer pour extraire une zone plus large autour du segment
            # Ici, on extrait avec 200ms de marge de part et d'autre
            audio_start_sec_buffered = max(0, audio_start_sec - SILENCE_DURATION)
            audio_end_sec_buffered = min(total_audio_duration, audio_end_sec + SILENCE_DURATION)

            # Nom du fichier segmenté
            output_filename = f"{nom_fichier_source}_{i}.wav"
            output_file = os.path.join(args.outdir, output_filename)

            # Utiliser sox pour extraire le segment brut avec buffer et forcer la conversion en mono
            tfm = sox.Transformer()
            tfm.channels(1)  # Forcer la sortie en mono
            tfm.trim(audio_start_sec_buffered, audio_end_sec_buffered)
            tfm.build_file(args.audio_filepath, output_file)
            
            # Post-traitement : détecter le début et la fin effectifs de la parole et ajouter 200ms de silence
            audio_data, sr = sf.read(output_file)
            if sr != SAMPLING_FREQ:
                print(f"Attention: taux d'échantillonnage {sr} différent de {SAMPLING_FREQ}.")
            # Détection du début et de la fin de la parole dans le segment extrait
            speech_start_idx, speech_end_idx = detect_speech(audio_data, sr)
            # Extraire la portion contenant la parole détectée
            speech_audio = audio_data[speech_start_idx:speech_end_idx+1]
            # Créer 200ms de silence (en échantillons) avec la même dimension que le signal (mono ici donc 1D)
            silence_samples = int(SILENCE_DURATION * sr)
            silence = np.zeros(silence_samples, dtype=audio_data.dtype)
            # Concaténer le silence avant et après la parole détectée
            processed_audio = np.concatenate([silence, speech_audio, silence])
            # Sauvegarder le segment final en mono
            sf.write(output_file, processed_audio, sr)
            
            # Écriture dans manifest.json
            sample = {
                "audio_start_sec": audio_start_sec_buffered,
                "audio_filepath": output_file,
                "duration": audio_end_sec_buffered - audio_start_sec_buffered,
                "text": t,
                "normalized_text": norm_transcripts[i],
                "uroman_tokens": tokens[i],
            }
            manifest_file.write(json.dumps(sample) + "\n")

            # Écriture dans metadata.csv
            csv_writer.writerow([output_filename, t])

    return segments, stride

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align and segment long audio files")
    parser.add_argument(
        "-a", "--audio_filepath", type=str, help="Path to input audio file"
    )
    parser.add_argument(
        "-t", "--text_filepath", type=str, help="Path to input text file"
    )
    parser.add_argument(
        "-l", "--lang", type=str, default="eng", help="ISO code of the language"
    )
    parser.add_argument(
        "-u", "--uroman_path", type=str, default="eng", help="Location to uroman/bin"
    )
    parser.add_argument(
        "-s",
        "--use_star",
        action="store_true",
        help="Use star at the start of transcript",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        help="Output directory to store segmented audio files",
    )
    print("Using torch version:", torch.__version__)
    print("Using torchaudio version:", torchaudio.__version__)
    print("Using device:", DEVICE)
    args = parser.parse_args()
    main(args)
