import os
import tempfile
import threading
import time
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yt_dlp
import whisper
import torch
import logging

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Global variables
active_jobs = {}
CACHED_MODEL = None

def get_whisper_model():
    """Load smallest Whisper model"""
    global CACHED_MODEL
    if CACHED_MODEL is None:
        # Use tiny model to save memory
        CACHED_MODEL = whisper.load_model("tiny")
        logging.info("Whisper tiny model loaded")
    return CACHED_MODEL

def download_audio(url, output_path, job_id):
    """Download audio from YouTube"""
    def progress_hook(d):
        if d['status'] == 'downloading':
            try:
                if '_percent_str' in d:
                    percent = float(d['_percent_str'].replace('%', ''))
                    active_jobs[job_id]['progress'] = percent * 0.4
                    active_jobs[job_id]['message'] = f"Downloading... {percent:.0f}%"
            except:
                active_jobs[job_id]['message'] = "Downloading audio..."
        elif d['status'] == 'finished':
            active_jobs[job_id]['progress'] = 40
            active_jobs[job_id]['message'] = "Starting AI transcription..."

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'progress_hooks': [progress_hook],
        'extractaudio': True,
        'audioformat': 'mp3',
        'audioquality': 5,  # Lower quality to save space
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Video')
            ydl.download([url])
        return title
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def transcribe_audio(audio_path, job_id):
    """Transcribe with lightweight Whisper"""
    try:
        active_jobs[job_id]['progress'] = 50
        active_jobs[job_id]['message'] = "Loading AI model..."
        
        model = get_whisper_model()
        
        active_jobs[job_id]['progress'] = 60
        active_jobs[job_id]['message'] = "AI transcribing..."
        
        # Transcribe
        result = model.transcribe(audio_path)
        
        active_jobs[job_id]['progress'] = 90
        active_jobs[job_id]['message'] = "Generating files..."
        
        return result['text'], result.get('segments', [])
        
    except Exception as e:
        raise Exception(f"Transcription failed: {str(e)}")

def process_transcription(job_id, url, languages, formats):
    """Main processing function"""
    try:
        active_jobs[job_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting...'
        }
        
        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        audio_path = os.path.join(temp_dir, 'audio.%(ext)s')
        
        # Download
        title = download_audio(url, audio_path, job_id)
        
        # Find downloaded file
        audio_file = None
        for file in os.listdir(temp_dir):
            if file.startswith('audio.'):
                audio_file = os.path.join(temp_dir, file)
                break
        
        if not audio_file:
            raise Exception("Audio file not found")
        
        # Transcribe
        text, segments = transcribe_audio(audio_file, job_id)
        
        # Generate files
        os.makedirs('transcripts', exist_ok=True)
        job_folder = os.path.join('transcripts', job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        files = {}
        for lang in languages:
            files[lang] = {}
            for fmt in formats:
                if fmt == 'txt':
                    content = f"Title: {title}\n\n{text}"
                elif fmt == 'srt':
                    content = "1\n00:00:00,000 --> 00:00:10,000\n" + text[:100] + "\n\n"
                elif fmt == 'json':
                    content = json.dumps({"title": title, "text": text}, indent=2)
                else:
                    content = text
                
                filename = f"{title[:20]}_{lang}.{fmt}"
                filepath = os.path.join(job_folder, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                files[lang][fmt] = filename
        
        # Cleanup
        try:
            os.remove(audio_file)
            os.rmdir(temp_dir)
        except:
            pass
        
        # Complete
        active_jobs[job_id].update({
            'status': 'completed',
            'progress': 100,
            'message': 'AI transcription complete!',
            'files': files,
            'title': title
        })
        
    except Exception as e:
        active_jobs[job_id] = {
            'status': 'error',
            'message': str(e)
        }

@app.route('/')
def index():
    return jsonify({
        "message": "YouTube Transcriber - LIGHTWEIGHT AI",
        "status": "running",
        "model": "whisper-tiny",
        "memory_optimized": True
    })

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        job_id = str(int(time.time() * 1000))
        
        data = request.get_json() if request.is_json else {
            'url': request.form.get('url'),
            'languages': json.loads(request.form.get('languages', '["en"]')),
            'formats': json.loads(request.form.get('formats', '["txt"]'))
        }
        
        url = data.get('url')
        languages = data.get('languages', ['en'])
        formats = data.get('formats', ['txt'])
        
        if not url:
            return jsonify({'success': False, 'error': 'No URL provided'}), 400
        
        # Start processing
        thread = threading.Thread(
            target=process_transcription,
            args=(job_id, url, languages, formats)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'Lightweight AI transcription started'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/progress/<job_id>')
def get_progress(job_id):
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(active_jobs[job_id])

@app.route('/download/<job_id>/<filename>')
def download_file(job_id, filename):
    try:
        file_path = os.path.join('transcripts', job_id, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting Lightweight AI Transcriber on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
