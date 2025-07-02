import os
import tempfile
import threading
import time
import json
import zipfile
from io import BytesIO
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import yt_dlp
from faster_whisper import WhisperModel
import torch
import logging
from werkzeug.utils import secure_filename

# Set up logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
UPLOAD_FOLDER = 'uploads'
TRANSCRIPTS_FOLDER = 'transcripts'

# Create directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TRANSCRIPTS_FOLDER, exist_ok=True)

# Global variables for progress tracking
active_jobs = {}

# Cache the Whisper model globally
CACHED_MODEL = None

def get_whisper_model():
    """Get cached faster-whisper model"""
    global CACHED_MODEL
    if CACHED_MODEL is None:
        start_time = time.time()
        
        # Determine device and compute type
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        else:
            device = "cpu"
            compute_type = "int8"
        
        # Load optimized model
        CACHED_MODEL = WhisperModel(
            "base",  # Small model for Railway's memory limits
            device=device,
            compute_type=compute_type,
            cpu_threads=2,
            num_workers=1
        )
        
        load_time = time.time() - start_time
        logging.info(f"Whisper model loaded in {load_time:.2f}s on {device}")
    
    return CACHED_MODEL

def download_audio(url, output_path, job_id):
    """Download audio from YouTube video"""
    def progress_hook(d):
        if d['status'] == 'downloading':
            try:
                if '_percent_str' in d:
                    percent_str = d['_percent_str'].replace('%', '')
                    percent = float(percent_str)
                    active_jobs[job_id]['progress'] = percent * 0.3  # 30% for download
                    active_jobs[job_id]['message'] = f"Downloading audio... {percent_str}%"
                else:
                    active_jobs[job_id]['message'] = "Downloading audio..."
            except:
                active_jobs[job_id]['message'] = "Downloading audio..."
        elif d['status'] == 'finished':
            active_jobs[job_id]['progress'] = 30
            active_jobs[job_id]['message'] = "Download complete, starting transcription..."

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path,
        'progress_hooks': [progress_hook],
        'extractaudio': True,
        'audioformat': 'wav',
        'audioquality': 192,
        'noplaylist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            active_jobs[job_id]['message'] = "Extracting video info..."
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown Video')
            duration = info.get('duration', 0)
            
            active_jobs[job_id]['message'] = f"Downloading: {title}"
            ydl.download([url])

        return title, duration
    except Exception as e:
        raise Exception(f"Download failed: {str(e)}")

def transcribe_audio(audio_path, job_id, source_language='auto'):
    """Transcribe audio using faster-whisper"""
    try:
        active_jobs[job_id]['progress'] = 35
        active_jobs[job_id]['message'] = "Loading Whisper AI model..."

        model = get_whisper_model()

        active_jobs[job_id]['progress'] = 40
        active_jobs[job_id]['message'] = "Starting AI transcription..."

        # Transcribe with faster-whisper
        transcription_start = time.time()
        
        language = None if source_language == 'auto' else source_language
        
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            language=language,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        # Process segments with progress updates
        transcript_segments = []
        segment_list = list(segments)
        total_segments = len(segment_list)
        
        for i, segment in enumerate(segment_list):
            transcript_segments.append({
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
            
            # Update progress (40% to 70%)
            progress = 40 + (i / max(total_segments, 1)) * 30
            active_jobs[job_id]['progress'] = progress
            active_jobs[job_id]['message'] = f"Transcribing... {i+1}/{total_segments} segments"

        transcription_time = time.time() - transcription_start
        
        # Update performance stats
        active_jobs[job_id]['performance_stats'] = {
            'transcription_time': f"{transcription_time:.2f}s",
            'language_detected': info.language,
            'language_probability': f"{info.language_probability:.2f}",
            'total_segments': total_segments
        }

        active_jobs[job_id]['progress'] = 70
        active_jobs[job_id]['message'] = "Transcription complete, generating files..."

        return transcript_segments, info

    except Exception as e:
        raise Exception(f"AI transcription failed: {str(e)}")

def generate_formats(segments, title, language_info, selected_formats, selected_languages):
    """Generate multiple output formats"""
    generated_files = {}
    
    # Base transcript text
    base_text = ' '.join([seg['text'] for seg in segments])
    
    for lang_code in selected_languages:
        lang_name = get_language_name(lang_code)
        generated_files[lang_code] = {}
        
        # For now, we'll use the same transcribed text for all languages
        # In a full implementation, you'd use translation APIs here
        text_content = base_text
        
        for format_type in selected_formats:
            if format_type == 'txt':
                content = f"YouTube Video: {title}\nLanguage: {lang_name}\nGenerated: {time.strftime('%Y-%m-%d %H:%M:%S')}\nPowered by Whisper AI\n\n{text_content}"
                generated_files[lang_code]['txt'] = content
                
            elif format_type == 'srt':
                srt_content = ""
                for i, seg in enumerate(segments, 1):
                    start_time = format_srt_time(seg['start'])
                    end_time = format_srt_time(seg['end'])
                    srt_content += f"{i}\n{start_time} --> {end_time}\n{seg['text']}\n\n"
                generated_files[lang_code]['srt'] = srt_content
                
            elif format_type == 'vtt':
                vtt_content = "WEBVTT\n\n"
                for seg in segments:
                    start_time = format_vtt_time(seg['start'])
                    end_time = format_vtt_time(seg['end'])
                    vtt_content += f"{start_time} --> {end_time}\n{seg['text']}\n\n"
                generated_files[lang_code]['vtt'] = vtt_content
                
            elif format_type == 'json':
                json_content = {
                    'title': title,
                    'language': lang_name,
                    'segments': segments,
                    'metadata': {
                        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                        'language_detected': language_info.language,
                        'confidence': language_info.language_probability,
                        'powered_by': 'Whisper AI'
                    }
                }
                generated_files[lang_code]['json'] = json.dumps(json_content, indent=2)
                
            elif format_type == 'csv':
                csv_content = "Start Time,End Time,Text\n"
                for seg in segments:
                    csv_content += f"{seg['start']:.2f},{seg['end']:.2f},\"{seg['text'].replace('\"', '\"\"')}\"\n"
                generated_files[lang_code]['csv'] = csv_content
                
            elif format_type == 'md':
                md_content = f"# {title}\n\n**Language:** {lang_name}\n**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n**Powered by:** Whisper AI\n\n## Transcript\n\n"
                for seg in segments:
                    md_content += f"**[{format_time(seg['start'])}]** {seg['text']}\n\n"
                generated_files[lang_code]['md'] = md_content
    
    return generated_files

def format_srt_time(seconds):
    """Format time for SRT files"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def format_vtt_time(seconds):
    """Format time for VTT files"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

def format_time(seconds):
    """Format time for display"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"

def get_language_name(lang_code):
    """Get language name from code"""
    language_map = {
        'en': 'English', 'es': 'Spanish', 'fr': 'French', 'de': 'German',
        'it': 'Italian', 'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese',
        'ja': 'Japanese', 'ko': 'Korean', 'ar': 'Arabic', 'hi': 'Hindi',
        'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish', 'sv': 'Swedish',
        'da': 'Danish', 'no': 'Norwegian', 'fi': 'Finnish', 'th': 'Thai',
        'vi': 'Vietnamese', 'id': 'Indonesian', 'ms': 'Malay', 'he': 'Hebrew',
        'uk': 'Ukrainian', 'cs': 'Czech', 'hu': 'Hungarian', 'ro': 'Romanian'
    }
    return language_map.get(lang_code, lang_code.upper())

def process_transcription(job_id, url=None, file_path=None, languages=None, formats=None, source_language='auto', options=None):
    """Main processing function with real AI"""
    try:
        active_jobs[job_id]['status'] = 'processing'
        active_jobs[job_id]['progress'] = 0
        active_jobs[job_id]['message'] = 'Starting YouTube transcription...'

        # Create temp directory
        temp_dir = tempfile.mkdtemp()
        
        if url:
            # Download from YouTube
            audio_path = os.path.join(temp_dir, 'audio.%(ext)s')
            title, duration = download_audio(url, audio_path, job_id)
            
            # Find downloaded file
            audio_file = None
            for file in os.listdir(temp_dir):
                if file.startswith('audio.'):
                    audio_file = os.path.join(temp_dir, file)
                    break
        else:
            # Use uploaded file
            audio_file = file_path
            title = "Uploaded Audio"

        if not audio_file or not os.path.exists(audio_file):
            raise Exception("Audio file not found")

        # Transcribe audio with AI
        segments, language_info = transcribe_audio(audio_file, job_id, source_language)
        
        # Generate multiple formats
        active_jobs[job_id]['progress'] = 80
        active_jobs[job_id]['message'] = "Generating output formats..."
        
        generated_files = generate_formats(segments, title, language_info, formats, languages)
        
        # Save files
        job_folder = os.path.join(TRANSCRIPTS_FOLDER, job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        file_paths = {}
        for lang_code, lang_files in generated_files.items():
            file_paths[lang_code] = {}
            for format_type, content in lang_files.items():
                safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
                filename = f"{safe_title}_{lang_code}.{format_type}"
                file_path = os.path.join(job_folder, filename)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                file_paths[lang_code][format_type] = filename

        # Cleanup temp files
        try:
            if url and os.path.exists(audio_file):
                os.remove(audio_file)
                os.rmdir(temp_dir)
        except:
            pass

        # Mark as completed
        active_jobs[job_id]['status'] = 'completed'
        active_jobs[job_id]['progress'] = 100
        active_jobs[job_id]['message'] = 'AI transcription completed!'
        active_jobs[job_id]['files'] = file_paths
        active_jobs[job_id]['title'] = title

    except Exception as e:
        active_jobs[job_id]['status'] = 'error'
        active_jobs[job_id]['message'] = str(e)
        logging.error(f"Transcription failed for job {job_id}: {str(e)}")

@app.route('/')
def index():
    return jsonify({
        "message": "YouTube Transcriber Backend - AI POWERED!",
        "status": "running",
        "whisper_model": "faster-whisper",
        "supported_languages": "99+",
        "ai_enabled": True,
        "endpoints": ["/transcribe", "/progress/<job_id>", "/download/<job_id>/<filename>", "/download-all/<job_id>"]
    })

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "ai_ready": True, "timestamp": time.time()})

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        # Generate unique job ID
        job_id = str(int(time.time() * 1000))
        
        # Initialize job tracking
        active_jobs[job_id] = {
            'status': 'initializing',
            'progress': 0,
            'message': 'Initializing AI transcription...',
            'performance_stats': {}
        }

        # Parse request data
        if request.is_json:
            data = request.get_json()
            url = data.get('url')
            languages = data.get('languages', ['en'])
            formats = data.get('formats', ['txt'])
            source_language = data.get('sourceLanguage', 'auto')
            options = data.get('options', {})
            file_path = None
        else:
            url = request.form.get('url')
            languages = json.loads(request.form.get('languages', '["en"]'))
            formats = json.loads(request.form.get('formats', '["txt"]'))
            source_language = request.form.get('sourceLanguage', 'auto')
            options = json.loads(request.form.get('options', '{}'))
            
            # Handle file upload
            file_path = None
            if 'file' in request.files:
                file = request.files['file']
                if file.filename != '':
                    filename = secure_filename(file.filename)
                    file_path = os.path.join(UPLOAD_FOLDER, f"{job_id}_{filename}")
                    file.save(file_path)

        if not url and not file_path:
            return jsonify({'success': False, 'error': 'No URL or file provided'}), 400

        # Start processing in background
        thread = threading.Thread(
            target=process_transcription,
            args=(job_id, url, file_path, languages, formats, source_language, options)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'AI transcription started'
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
        file_path = os.path.join(TRANSCRIPTS_FOLDER, job_id, filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
        
        return send_file(file_path, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-all/<job_id>')
def download_all(job_id):
    try:
        if job_id not in active_jobs or active_jobs[job_id]['status'] != 'completed':
            return jsonify({'error': 'Job not completed'}), 404

        job_folder = os.path.join(TRANSCRIPTS_FOLDER, job_id)
        if not os.path.exists(job_folder):
            return jsonify({'error': 'Files not found'}), 404

        # Create ZIP file
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(job_folder):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, job_folder)
                    zf.write(file_path, arcname)

        memory_file.seek(0)
        
        title = active_jobs[job_id].get('title', 'transcripts')
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).rstrip()[:30]
        zip_filename = f"{safe_title}_all_formats.zip"

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name=zip_filename
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🤖 Starting AI-Powered YouTube Transcriber on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
