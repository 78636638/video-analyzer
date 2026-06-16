#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import subprocess
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename

# Reuse the analyzer's Config loader so the UI prefill stays in sync with
# whatever the CLI would actually use (user config > default config).
try:
    from video_analyzer.config import Config as _AnalyzerConfig
    _HAVE_ANALYZER_CONFIG = True
    _CONFIG_LOAD_ERROR = None
except Exception as _import_err:  # pragma: no cover - defensive guard
    _AnalyzerConfig = None
    _HAVE_ANALYZER_CONFIG = False
    _CONFIG_LOAD_ERROR = str(_import_err)

# Initialize logger
logger = logging.getLogger(__name__)

class VideoAnalyzerUI:
    def __init__(self, host='localhost', port=5000, dev_mode=False):
        self.app = Flask(__name__)
        self.host = host
        self.port = port
        self.dev_mode = dev_mode
        self.sessions = {}
        
        # Ensure tmp directories exist
        self.tmp_root = Path(tempfile.gettempdir()) / 'video-analyzer-ui'
        self.uploads_dir = self.tmp_root / 'uploads'
        self.results_dir = self.tmp_root / 'results'
        self.audit_output_root = Path('output')
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.audit_output_root.mkdir(parents=True, exist_ok=True)
        self.heartbeat_interval = 10
        
        self.setup_routes()

    @staticmethod
    def _create_analysis_state() -> dict:
        return {
            'state': 'idle',
            'step': 'waiting',
            'step_label': 'Waiting to start',
            'events': [],
            'event_seq': 0,
            'error': None,
            'last_log': '',
            'total_frames': None,
            'analyzed_frames': 0,
            'started_at': None,
            'finished_at': None,
            'process_id': None,
        }

    @staticmethod
    def _is_video_file(path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in {'.mp4', '.avi', '.mov', '.mkv'}

    def _resolve_analysis_file(self, session: dict) -> Path:
        """Resolve the real analysis.json location for a session.

        The CLI should write directly into the session output directory, but we
        keep a compatibility fallback for older runs that still wrote to the
        default `output/analysis.json`.
        """
        results_dir = Path(session['results_dir'])
        logger.debug("Looking for analysis file in: %s", results_dir)

        if not results_dir.exists():
            raise FileNotFoundError(f"Results directory not found: {results_dir}")

        analysis_file = results_dir / 'analysis.json'
        default_output = Path('output/analysis.json')

        if default_output.exists() and not analysis_file.exists():
            logger.debug("Found analysis file in legacy output directory: %s", default_output)
            try:
                default_output.rename(analysis_file)
                logger.debug("Moved legacy analysis file to: %s", analysis_file)
            except Exception as move_error:
                logger.warning("Could not move legacy analysis file, trying copy fallback: %s", move_error)
                analysis_file.write_text(default_output.read_text(encoding='utf-8'), encoding='utf-8')
                default_output.unlink()

        if not analysis_file.exists():
            raise FileNotFoundError(f"Analysis file not found: {analysis_file}")

        return analysis_file

    def _get_audit_output_dir(self, session_id: str) -> Path:
        return self.audit_output_root / session_id

    def _persist_results_for_audit(self, session_id: str, session: dict) -> Path:
        """Keep a reviewable copy of UI analysis outputs under ./output."""
        source_dir = Path(session['results_dir'])
        audit_dir = self._get_audit_output_dir(session_id)
        audit_dir.parent.mkdir(parents=True, exist_ok=True)

        if audit_dir.exists():
            shutil.rmtree(audit_dir)
        shutil.copytree(source_dir, audit_dir)
        session['audit_output_dir'] = str(audit_dir)
        logger.info("Session %s persisted audit outputs to %s", session_id, audit_dir)
        return audit_dir

    def _create_session(self, video_path: Path, filename: str, owns_upload: bool) -> dict:
        session_id = str(uuid.uuid4())
        session_results_dir = self.results_dir / session_id
        session_results_dir.mkdir(parents=True, exist_ok=True)
        session = {
            'video_path': str(video_path),
            'results_dir': str(session_results_dir),
            'filename': filename,
            'owns_upload': owns_upload,
            'lock': threading.RLock(),
            'analysis': self._create_analysis_state(),
        }
        self.sessions[session_id] = session
        return {
            'session_id': session_id,
            'message': 'Session created successfully',
            'video_path': str(video_path),
            'filename': filename,
        }

    @staticmethod
    def _session_expired_response() -> tuple:
        return jsonify({
            'error': 'Session expired. Please re-select or upload the video again.',
            'code': 'session_expired',
        }), 410

    def _append_event(self, session: dict, event_type: str, payload: dict) -> dict:
        with session['lock']:
            analysis = session['analysis']
            analysis['event_seq'] += 1
            event = {
                'id': analysis['event_seq'],
                'type': event_type,
                'payload': payload,
                'timestamp': time.time(),
            }
            analysis['events'].append(event)
            if len(analysis['events']) > 5000:
                analysis['events'] = analysis['events'][-5000:]
            return event

    def _build_status_payload(self, session: dict) -> dict:
        with session['lock']:
            analysis = session['analysis']
            total_frames = analysis['total_frames']
            analyzed_frames = analysis['analyzed_frames']
            progress_percent = 0
            if total_frames:
                progress_percent = round((analyzed_frames / total_frames) * 100, 2)

            results_ready = Path(session['results_dir'], 'analysis.json').exists()
            return {
                'state': analysis['state'],
                'step': analysis['step'],
                'step_label': analysis['step_label'],
                'error': analysis['error'],
                'last_log': analysis['last_log'],
                'total_frames': total_frames,
                'analyzed_frames': analyzed_frames,
                'progress_percent': progress_percent,
                'started_at': analysis['started_at'],
                'finished_at': analysis['finished_at'],
                'process_id': analysis['process_id'],
                'results_ready': results_ready,
                'output_dir': session.get('output_dir'),
                'video_path': session.get('video_path'),
            }

    def _emit_status(self, session: dict, **updates) -> None:
        with session['lock']:
            analysis = session['analysis']
            analysis.update(updates)
        self._append_event(session, 'status', self._build_status_payload(session))

    def _append_log_line(self, session: dict, line: str) -> None:
        self._update_progress_from_line(session, line)
        with session['lock']:
            session['analysis']['last_log'] = line
        self._append_event(session, 'log', {'line': line})

    def _update_progress_from_line(self, session: dict, line: str) -> None:
        status_updates = None
        extracted_match = re.search(r'Extracted (\d+) frames from video', line)
        analyzed_match = re.search(r'Successfully analyzed frame (\d+)', line)
        progress_match = re.search(r'Frame analysis progress:\s*(\d+)/(\d+)', line)

        if 'Initiation Input:' in line:
            status_updates = {'step': 'audio_init', 'step_label': 'Initializing audio model'}
        elif 'Extracting audio from video' in line:
            status_updates = {'step': 'extracting_audio', 'step_label': 'Extracting audio'}
        elif 'Transcribing audio' in line or 'Processing audio with duration' in line:
            status_updates = {'step': 'transcribing_audio', 'step_label': 'Transcribing audio'}
        elif 'Extracting frames from video' in line:
            status_updates = {'step': 'extracting_frames', 'step_label': 'Extracting frames'}
        elif extracted_match:
            status_updates = {
                'step': 'frames_extracted',
                'step_label': 'Frames extracted',
                'total_frames': int(extracted_match.group(1)),
            }
        elif 'Analyzing frames...' in line:
            status_updates = {'step': 'analyzing_frames', 'step_label': 'Analyzing frames'}
        elif progress_match:
            analyzed_frames = int(progress_match.group(1))
            total_frames = int(progress_match.group(2))
            status_updates = {
                'step': 'analyzing_frames',
                'step_label': 'Analyzing frames',
                'analyzed_frames': analyzed_frames,
                'total_frames': total_frames,
            }
        elif analyzed_match:
            analyzed_frames = int(analyzed_match.group(1)) + 1
            status_updates = {
                'step': 'analyzing_frames',
                'step_label': 'Analyzing frames',
                'analyzed_frames': analyzed_frames,
            }
        elif 'Reconstructing video description' in line:
            total_frames = session['analysis'].get('total_frames')
            status_updates = {
                'step': 'reconstructing_video',
                'step_label': 'Reconstructing video description',
                'analyzed_frames': total_frames or session['analysis'].get('analyzed_frames', 0),
            }
        elif 'Successfully reconstructed video description' in line:
            status_updates = {'step': 'reconstruction_complete', 'step_label': 'Video reconstruction complete'}
        elif 'Analysis complete.' in line:
            total_frames = session['analysis'].get('total_frames')
            status_updates = {
                'state': 'completed',
                'step': 'completed',
                'step_label': 'Analysis completed',
                'analyzed_frames': total_frames or session['analysis'].get('analyzed_frames', 0),
                'finished_at': time.time(),
            }
        elif 'Analysis failed' in line:
            status_updates = {
                'state': 'failed',
                'step': 'failed',
                'step_label': 'Analysis failed',
                'error': line,
                'finished_at': time.time(),
            }

        if status_updates:
            self._emit_status(session, **status_updates)

    def _run_analysis_subprocess(self, session_id: str) -> None:
        session = self.sessions.get(session_id)
        if not session:
            return

        try:
            process = subprocess.Popen(
                session['cmd'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                env=os.environ.copy(),
            )
            logger.info("Session %s subprocess started with pid %s", session_id, process.pid)
            self._emit_status(
                session,
                state='running',
                step='starting',
                step_label='Starting analysis process',
                process_id=process.pid,
                started_at=time.time(),
                finished_at=None,
                error=None,
                total_frames=None,
                analyzed_frames=0,
            )

            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                logger.debug("Session %s output: %s", session_id, line)
                self._append_log_line(session, line)

            process.wait()
            if process.returncode == 0:
                try:
                    audit_dir = self._persist_results_for_audit(session_id, session)
                    self._append_event(session, 'log', {'line': f'Audit outputs saved to {audit_dir}'})
                except Exception as persist_error:
                    logger.error("Session %s failed to persist audit outputs: %s", session_id, persist_error)
                    self._append_event(session, 'log', {'line': f'Failed to persist audit outputs: {persist_error}'})
                self._emit_status(
                    session,
                    state='completed',
                    step='completed',
                    step_label='Analysis completed',
                    finished_at=time.time(),
                )
                self._append_event(session, 'log', {'line': 'Analysis completed successfully'})
            else:
                failure_message = f'Analysis failed with code {process.returncode}'
                self._emit_status(
                    session,
                    state='failed',
                    step='failed',
                    step_label='Analysis failed',
                    error=failure_message,
                    finished_at=time.time(),
                )
                self._append_event(session, 'log', {'line': failure_message})
        except Exception as e:
            logger.error("Session %s analysis error: %s", session_id, e)
            self._emit_status(
                session,
                state='failed',
                step='failed',
                step_label='Analysis failed',
                error=str(e),
                finished_at=time.time(),
            )
            self._append_event(session, 'log', {'line': f'Error during analysis: {str(e)}'})
        finally:
            self._append_event(session, 'status', self._build_status_payload(session))

    def _list_saved_videos(self) -> list[dict]:
        videos = []
        for session_upload_dir in sorted(self.uploads_dir.iterdir(), reverse=True):
            if not session_upload_dir.is_dir():
                continue

            for file_path in sorted(session_upload_dir.iterdir(), reverse=True):
                if not self._is_video_file(file_path):
                    continue

                stat = file_path.stat()
                videos.append({
                    'video_id': session_upload_dir.name,
                    'filename': file_path.name,
                    'video_path': str(file_path),
                    'size_bytes': stat.st_size,
                    'modified_at': stat.st_mtime,
                })
        return videos
        
    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('index.html')

        @self.app.route('/api/config', methods=['GET'])
        def get_client_config():
            """Expose the `clients` node of the analyzer config to the UI
            so the form fields can be auto-filled on page load.
            """
            if not _HAVE_ANALYZER_CONFIG:
                logger.warning(
                    "video_analyzer package not importable; returning empty config: %s",
                    _CONFIG_LOAD_ERROR,
                )
                return jsonify({
                    'available': False,
                    'error': _CONFIG_LOAD_ERROR or 'video_analyzer not available',
                    'clients': {},
                    'analysis': {},
                })

            try:
                # Match the CLI default: look in ./config, fall back to the
                # packaged default. We never mutate the config here.
                config = _AnalyzerConfig(config_dir='config')
                clients = config.get('clients', {}) or {}
                analysis = config.get('analysis', {}) or {}
                return jsonify({
                    'available': True,
                    'clients': clients,
                    'analysis': analysis,
                })
            except Exception as e:
                logger.error("Failed to load analyzer config for UI: %s", e)
                return jsonify({
                    'available': False,
                    'error': str(e),
                    'clients': {},
                    'analysis': {},
                }), 500

        @self.app.route('/videos', methods=['GET'])
        def list_videos():
            return jsonify({'videos': self._list_saved_videos()})

        @self.app.route('/videos/select', methods=['POST'])
        def select_video():
            payload = request.get_json(silent=True) or {}
            video_id = payload.get('video_id')
            if not video_id:
                return jsonify({'error': 'Missing video_id'}), 400

            upload_dir = self.uploads_dir / secure_filename(video_id)
            if not upload_dir.exists() or not upload_dir.is_dir():
                return jsonify({'error': 'Selected video not found'}), 404

            video_file = next((path for path in sorted(upload_dir.iterdir()) if self._is_video_file(path)), None)
            if video_file is None:
                return jsonify({'error': 'Selected video file not found'}), 404

            response = self._create_session(video_file, video_file.name, owns_upload=False)
            response['message'] = 'Existing video selected successfully'
            return jsonify(response)

        @self.app.route('/videos/<video_id>', methods=['DELETE'])
        def delete_video(video_id):
            safe_video_id = secure_filename(video_id)
            upload_dir = self.uploads_dir / safe_video_id
            results_dir = self.results_dir / safe_video_id
            if not upload_dir.exists() or not upload_dir.is_dir():
                return jsonify({'error': 'Video not found'}), 404

            try:
                for file_path in upload_dir.glob('*'):
                    if file_path.is_file():
                        file_path.unlink()
                upload_dir.rmdir()

                if results_dir.exists():
                    for file_path in results_dir.glob('**/*'):
                        if file_path.is_file():
                            file_path.unlink()
                    for dir_path in sorted(results_dir.glob('**/*'), reverse=True):
                        if dir_path.is_dir():
                            dir_path.rmdir()
                    results_dir.rmdir()

                for session_id, session in list(self.sessions.items()):
                    if Path(session['video_path']).parent.name == safe_video_id:
                        del self.sessions[session_id]

                return jsonify({'message': 'Video deleted successfully', 'videos': self._list_saved_videos()})
            except Exception as e:
                logger.error(f"Delete video error: {e}")
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/upload', methods=['POST'])
        def upload_file():
            if 'video' not in request.files:
                return jsonify({'error': 'No video file provided'}), 400
                
            file = request.files['video']
            if file.filename == '':
                return jsonify({'error': 'No selected file'}), 400
                
            if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return jsonify({'error': 'Invalid file type'}), 400
                
            try:
                # Create upload directory tied to an owned session
                upload_id = str(uuid.uuid4())
                session_upload_dir = self.uploads_dir / upload_id
                session_upload_dir.mkdir(parents=True)
                # Save file
                filename = secure_filename(file.filename)
                filepath = session_upload_dir / filename
                file.save(filepath)
                response = self._create_session(filepath, filename, owns_upload=True)
                response['video_id'] = upload_id
                return jsonify(response)
            except Exception as e:
                logger.error(f"Upload error: {e}")
                return jsonify({'error': str(e)}), 500
                
        @self.app.route('/analyze/<session_id>', methods=['POST'])
        def analyze(session_id):
            if session_id not in self.sessions:
                return self._session_expired_response()
                
            session = self.sessions[session_id]
            
            # Build command
            cmd = ['video-analyzer', session['video_path']]
            
            # Add optional parameters
            for param, value in request.form.items():
                if value:  # Only add parameters with values
                    if param in ['keep-frames', 'dev']:  # Flags without values
                        cmd.append(f'--{param}')
                    else:
                        cmd.extend([f'--{param}', value])
                        
            # Create output directory if it doesn't exist
            results_dir = Path(session['results_dir'])
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Add output directory
            cmd.extend(['--output', str(results_dir)])
            
            # Store output directory in session for later use
            session['output_dir'] = str(results_dir)
            logger.debug(f"Set output directory to: {results_dir}")
            
            # Reset and store analysis state for background processing
            with session['lock']:
                session['analysis'] = self._create_analysis_state()
            session['cmd'] = cmd
            logger.info(
                "Session %s queued analysis for %s with output %s",
                session_id,
                session.get('video_path'),
                session.get('output_dir'),
            )
            self._emit_status(session, state='queued', step='queued', step_label='Queued for analysis')

            worker = threading.Thread(
                target=self._run_analysis_subprocess,
                args=(session_id,),
                daemon=True,
            )
            session['worker'] = worker
            worker.start()
            
            return jsonify({
                'message': 'Analysis started',
                'status': self._build_status_payload(session),
            })

        @self.app.route('/analyze/<session_id>/status')
        def get_analysis_status(session_id):
            if session_id not in self.sessions:
                return self._session_expired_response()

            session = self.sessions[session_id]
            last_event_id = int(request.args.get('last_event_id', 0))
            with session['lock']:
                events = [
                    event for event in session['analysis']['events']
                    if event['id'] > last_event_id
                ]

            return jsonify({
                'status': self._build_status_payload(session),
                'events': events,
            })
            
        @self.app.route('/analyze/<session_id>/stream')
        def stream_output(session_id):
            if session_id not in self.sessions:
                return self._session_expired_response()
                
            session = self.sessions[session_id]
            if 'cmd' not in session:
                return jsonify({'error': 'Analysis not started'}), 400

            try:
                last_event_id = int(request.args.get('last_event_id') or request.headers.get('Last-Event-ID') or 0)
            except ValueError:
                last_event_id = 0
                
            def generate_output():
                logger.info("Session %s SSE stream opened", session_id)
                yield "retry: 3000\n\n"
                current_event_id = last_event_id
                last_heartbeat = time.time()
                last_status_signature = None
                try:
                    while True:
                        with session['lock']:
                            events = [
                                event for event in session['analysis']['events']
                                if event['id'] > current_event_id
                            ]

                        for event in events:
                            current_event_id = event['id']
                            payload = json.dumps(event['payload'], ensure_ascii=False)
                            yield f"id: {event['id']}\nevent: {event['type']}\ndata: {payload}\n\n"

                        status_payload = self._build_status_payload(session)
                        status_signature = json.dumps(status_payload, sort_keys=True, ensure_ascii=False)
                        if status_signature != last_status_signature:
                            last_status_signature = status_signature
                            yield f"event: status\ndata: {json.dumps(status_payload, ensure_ascii=False)}\n\n"

                        if time.time() - last_heartbeat >= self.heartbeat_interval:
                            last_heartbeat = time.time()
                            yield f"event: heartbeat\ndata: {json.dumps(status_payload, ensure_ascii=False)}\n\n"

                        if status_payload['state'] in {'completed', 'failed'} and not events:
                            break

                        time.sleep(1)
                finally:
                    logger.info("Session %s SSE stream closed", session_id)
                    
            return Response(
                generate_output(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no',
                }
            )
            
        @self.app.route('/results/<session_id>')
        def get_results(session_id):
            if session_id not in self.sessions:
                return self._session_expired_response()
                
            session = self.sessions[session_id]
            try:
                analysis_file = self._resolve_analysis_file(session)
            except FileNotFoundError as e:
                logger.error(str(e))
                return jsonify({'error': str(e)}), 404
            except Exception as e:
                logger.error("Error resolving analysis file: %s", e)
                return jsonify({'error': f'Error accessing analysis file: {str(e)}'}), 500
                
            try:
                return send_file(
                    analysis_file,
                    mimetype='application/json',
                    as_attachment=True,
                    download_name=f"analysis_{session['filename']}.json"
                )
            except Exception as e:
                logger.error(f"Error sending file: {e}")
                return jsonify({'error': f'Error sending file: {str(e)}'}), 500

        @self.app.route('/results/<session_id>/preview')
        def get_results_preview(session_id):
            if session_id not in self.sessions:
                return self._session_expired_response()

            session = self.sessions[session_id]
            try:
                analysis_file = self._resolve_analysis_file(session)
                with analysis_file.open(encoding='utf-8') as f:
                    payload = json.load(f)
                return jsonify(payload)
            except FileNotFoundError as e:
                logger.error(str(e))
                return jsonify({'error': str(e)}), 404
            except json.JSONDecodeError as e:
                logger.error("Invalid analysis JSON in %s: %s", session.get('results_dir'), e)
                return jsonify({'error': f'Invalid analysis JSON: {str(e)}'}), 500
            except Exception as e:
                logger.error("Error returning analysis preview: %s", e)
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/cleanup/<session_id>', methods=['POST'])
        def cleanup_session(session_id):
            if session_id not in self.sessions:
                return self._session_expired_response()
                
            try:
                session = self.sessions[session_id]
                # Clean up upload directory only when this session owns the uploaded file
                if session.get('owns_upload'):
                    upload_dir = Path(session['video_path']).parent
                    if upload_dir.exists():
                        for file in upload_dir.glob('*'):
                            file.unlink()
                        upload_dir.rmdir()
                
                # Clean up results directory
                results_dir = Path(session['results_dir'])
                if results_dir.exists():
                    for file in results_dir.glob('**/*'):
                        if file.is_file():
                            file.unlink()
                    for dir_path in sorted(results_dir.glob('**/*'), reverse=True):
                        if dir_path.is_dir():
                            dir_path.rmdir()
                    results_dir.rmdir()
                
                process_id = session.get('analysis', {}).get('process_id')
                if process_id:
                    try:
                        os.kill(process_id, 15)
                    except OSError:
                        pass
                
                del self.sessions[session_id]
                return jsonify({'message': 'Session cleaned up successfully'})
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                return jsonify({'error': str(e)}), 500
    
    def run(self):
        self.app.run(
            host=self.host,
            port=self.port,
            debug=self.dev_mode
        )

def main():
    parser = argparse.ArgumentParser(description="Video Analyzer UI Server")
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to listen on')
    parser.add_argument('--dev', action='store_true', help='Enable development mode')
    parser.add_argument('--log-file', help='Log file path')
    
    args = parser.parse_args()
    
    # Configure logging
    log_config = {
        'level': logging.DEBUG if args.dev else logging.INFO,
        'format': '%(asctime)s - %(levelname)s - %(message)s',
    }
    if args.log_file:
        log_config['filename'] = args.log_file
    logging.basicConfig(**log_config)
    
    try:
        # Check if video-analyzer is installed
        subprocess.run(['video-analyzer', '--help'], capture_output=True, check=True)
    except subprocess.CalledProcessError:
        logger.error("video-analyzer command not found. Please install video-analyzer package.")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("video-analyzer command not found. Please install video-analyzer package.")
        sys.exit(1)
    
    # Start server
    server = VideoAnalyzerUI(
        host=args.host,
        port=args.port,
        dev_mode=args.dev
    )
    server.run()

if __name__ == '__main__':
    main()
